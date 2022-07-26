from hashlib import new
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from functools import reduce
import operator
from timm.models.layers import DropPath, trunc_normal_

from mmcv.runner import load_checkpoint
from mmaction.utils import get_root_logger
from ..builder import BACKBONES

from functools import reduce, lru_cache
from operator import mul
from einops import rearrange

from .ops.modules import MS3DDeformAttn
from .swin_transformer import (
    Mlp, window_partition, window_reverse, get_window_size, PatchEmbed3D,
    PatchMerging)


@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    """Attention mask for EACH WINDOW"""
    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in [slice(-window_size[0]),
              slice(-window_size[0],
              -shift_size[0]),
              slice(-shift_size[0], None)]:
        for h in [slice(-window_size[1]),
                  slice(-window_size[1],
                  -shift_size[1]),
                  slice(-shift_size[1], None)]:
            for w in [slice(-window_size[2]),
                      slice(-window_size[2],
                      -shift_size[2]),
                      slice(-shift_size[2], None)]:
                img_mask[:, d, h, w, :] = cnt
                cnt += 1

    # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]

    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(
        attn_mask != 0, float(0.0)).masked_fill(attn_mask == 0, float(1.0)
    )  # nW, ws[0]*ws[1]*ws[2], ws[0]*ws[1]*ws[2]
    D_, H_, W_ = window_size
    attn_mask = attn_mask.view(-1, H_ * W_ * D_, D_, H_, W_)
    return attn_mask


class PositionEmbeddingSine(nn.Module):
    """ I believe this is correct, but it would be worth to check it (again)
    TODO
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, mask):
        assert mask is not None
        z_embed = mask.cumsum(2, dtype=torch.float32) * mask / math.pi
        y_embed = mask.cumsum(3, dtype=torch.float32) * mask / math.pi
        x_embed = mask.cumsum(4, dtype=torch.float32) * mask / math.pi
        embed = mask * (x_embed.sin() + (math.pi * y_embed).sin() + z_embed)
        embed = embed.flatten(2)
        return embed, torch.diagonal(embed, dim1=1, dim2=2)


class WindowAttention3D(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative
    position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of
                                  the window.
        num_heads (int): Number of attention heads.
        num_points (int | None, optional): Number of points to sample for the
                                           attention.
        attn_drop (float, optional): Dropout ratio of attention weight.
                                     Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        num_points=4,
        attn_drop=0.0,
        proj_drop=0.0,
    ):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads

        self.self_attn = MS3DDeformAttn(
            dim, num_heads, num_points, reduce(operator.mul, window_size, 1)) # math.prod(window_size)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.pos_embedd = PositionEmbeddingSine()
        self.norm = nn.LayerNorm(dim)

    @staticmethod
    def with_pos_embed(tensor, pos):
        if pos is None:
            return tensor
        Nw, S = pos.shape
        _, _, D = tensor.shape
        tensor_ = tensor.view(-1, Nw, S, D)
        return (tensor_ + pos[None, :, :, None]).view(tensor.shape)

    @staticmethod
    def get_reference_points(spatial_shapes, device):
        # Get reference points from all the video.
        D_, H_, W_ = spatial_shapes
        ref_z, ref_y, ref_x = torch.meshgrid(
            torch.linspace(
                0.5, D_ - 0.5, D_, dtype=torch.float32, device=device),
            torch.linspace(
                0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(
                0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_z = ref_z.reshape(-1)[None] / D_
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_

        # 1, DxHxW, 3
        reference_points = torch.stack((ref_x, ref_y, ref_z), -1)
        return reference_points

    def get_valid_ratio(self, mask, shape):
        # Not gonna need this.
        D, H, W = shape
        mask = mask.view(-1, *shape)
        valid_D = torch.sum(1 - mask[:, :, 0, 0], 1)
        valid_H = torch.sum(1 - mask[:, 0, :, 0], 1)
        valid_W = torch.sum(1 - mask[:, 0, 0, :], 1)
        valid_ratio_d = valid_D.float() / D
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack(
            [valid_ratio_w, valid_ratio_h, valid_ratio_d], -1)  # DxHxW, 3
        return valid_ratio

    def forward(self, x, mask=None):
        """Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        # The center of every patch is a point
        reference_points = self.get_reference_points(
            self.window_size, device=x.device)
        matrix_embed, embed = self.pos_embedd(x, mask)
        # attn
        attn = self.self_attn(
            self.with_pos_embed(x, embed),
            reference_points,
            x,
            self.window_size,
            0,  # level_start_index
            mask,
            matrix_embed
        )

        attn = self.attn_drop(attn)
        x = self.proj_drop(self.proj(self.norm(x + attn)))
        return x


class SwinTransformerBlock3D(nn.Module):
    """Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        num_points (int | None, optional): Number of points to sample for the
                                           attention.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.
                                          Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        num_heads,
        window_size=(2, 7, 7),
        shift_size=(0, 0, 0),
        mlp_ratio=4.0,
        num_points=4,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_checkpoint=False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        assert (
            0 <= self.shift_size[0] < self.window_size[0]
        ), "shift_size must in 0-window_size"
        assert (
            0 <= self.shift_size[1] < self.window_size[1]
        ), "shift_size must in 0-window_size"
        assert (
            0 <= self.shift_size[2] < self.window_size[2]
        ), "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            num_points=num_points,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size(
            (D, H, W), self.window_size, self.shift_size
        )

        x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(
                x,
                shifts=(-shift_size[0], -shift_size[1], -shift_size[2]),
                dims=(1, 2, 3),
            )
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = mask_matrix
        # partition windows  (B*nW, Wd*Wh*Ww, C)
        x_windows = window_partition(shifted_x, window_size)
        # W-MSA/SW-MSA  (B*nW, Wd*Wh*Ww, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(
            attn_windows, window_size, B, Dp, Hp, Wp
        )  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(
                shifted_x,
                shifts=(shift_size[0], shift_size[1], shift_size[2]),
                dims=(1, 2, 3),
            )
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        """Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """

        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (1,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
                           Default: 4.
        num_points (int | None, optional): Number of points to sample for the
                                           attention.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate.
                                                    Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer.
                                          Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the
                                                 end of the layer.
                                                 Default: None
    """

    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size=(1, 7, 7),
        mlp_ratio=4.0,
        num_points=4,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock3D(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    num_points=num_points,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                )
                for i in range(depth)
            ]
        )

        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)

    def forward(self, x):
        """Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size(
            (D, H, W), self.window_size, self.shift_size
        )
        x = rearrange(x, "b c d h w -> b d h w c")
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(B, D, H, W, -1)

        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, "b d h w c -> b c d h w")
        return x

@BACKBONES.register_module()
class DeformableSwinTransformer3D(nn.Module):
    """Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer
        using Shifted Windows`  -  https://arxiv.org/pdf/2103.14030

    Args:
        patch_size (int | tuple(int)): Patch size. Default: (4,4,4).
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels.
                         Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
                           Default: 4.
        num_points (int | None, optional): Number of points to sample for the
                                           attention.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer: Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding.
                           Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
    """

    def __init__(
        self,
        pretrained=None,
        pretrained2d=True,
        clap=False,
        patch_size=(4, 4, 4),
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=(2, 7, 7),
        mlp_ratio=4.0,
        num_points=4,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        patch_norm=False,
        frozen_stages=-1,
        use_checkpoint=False,
        qkv_bias=True,  # Cann't delete this because of the swin config
        qk_scale=None,  # Cann't delete this because of the swin config
    ):
        super().__init__()

        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.clap = clap
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                num_points=num_points,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    sum(depths[:i_layer]): sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer < self.num_layers - 1 else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        # add a norm layer for each output
        self.norm = norm_layer(self.num_features)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def inflate_weights(self, logger):
        """Inflate the swin2d parameters to swin3d.

        The differences between swin3d and swin2d mainly lie in an extra
        axis. To utilize the pretrained parameters in 2d model,
        the weight of swin2d models should be inflated to fit in the shapes of
        the 3d counterpart.

        Args:
            logger (logging.Logger): The logger used to print
                debugging infomation.
        """
        checkpoint = torch.load(self.pretrained, map_location="cpu")
        state_dict = checkpoint["model"]

        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [
            k for k in state_dict.keys() if "relative_position_index" in k
        ]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]

        state_dict["patch_embed.proj.weight"] = (
            state_dict["patch_embed.proj.weight"]
            .unsqueeze(2)
            .repeat(1, 1, self.patch_size[0], 1, 1)
            / self.patch_size[0]
        )

        msg = self.load_state_dict(state_dict, strict=False)
        logger.info(msg)
        logger.info(f"=> loaded successfully '{self.pretrained}'")
        del checkpoint
        torch.cuda.empty_cache()

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            logger.info(f"load model from: {self.pretrained}")

            if self.pretrained2d:
                # Inflate 2D model into 3D model.
                self.inflate_weights(logger)
            elif self.clap:
                state_dict = torch.load(self.pretrained)['model_state']
                for key in list(state_dict.keys()):
                    if 'video_encoder' in key:
                        state_dict[key[21:]] = state_dict.pop(key)
                    else:
                        del state_dict[key]
                msg = self.load_state_dict(state_dict, strict=False)
                logger.info(msg)
                logger.info(f"=> loaded successfully '{self.pretrained}'")
                del state_dict
                torch.cuda.empty_cache()
            else:
                # Directly load 3D model.
                load_checkpoint(
                    self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError("pretrained must be a str or None")

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)

        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x.contiguous())

        x = rearrange(x, "n c d h w -> n d h w c")
        x = self.norm(x)
        x = rearrange(x, "n d h w c -> n c d h w")

        return x

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(DeformableSwinTransformer3D, self).train(mode)
        self._freeze_stages()
