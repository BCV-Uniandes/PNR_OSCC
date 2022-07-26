# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from
# https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..functions import MS3DDeformAttnFunction


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(
            "invalid input for _is_power_of_2: {} (type: {})".format(
                n, type(n))
            )
    return (n & (n-1) == 0) and n != 0


class MS3DDeformAttn(nn.Module):
    def __init__(self, d_model=128, n_heads=8, n_points=4, n_windows=128):
        """
        Multi-Scale 3D Deformable Attention Module
        :param d_model      hidden dimension
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head
                            per feature level
        :param n_windows    number of windows (swin)
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                'd_model must be divisible by n_heads, but got {} and {}'
                .format(d_model, n_heads))
        _d_per_head = d_model // n_heads

        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MS3DDeformAttn to make "
                          "the dimension of the attention heads a power of 2. "
                          "It is more efficient in our CUDA implementation.")

        # self.im2col_step = 64  # WHAT IS THIS?!?!?! -> B_ % im2col == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_points = n_points
        self.n_levels = 1

        self.query_map = nn.Linear(d_model, 1)
        self.sampling_offsets = nn.Linear(
            n_windows + 1, n_heads * n_points * 3)
        self.attention_weights = nn.Linear(
            n_windows + 1, n_heads * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(
            self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.sin(), thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 3)
            .repeat(1, 1, self.n_points, 1)
        )  # H, L, P, 3

        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1

        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten,
                input_spatial_shapes, input_level_start_index,
                mask=None, pos_embed=None):
        """
        B_ = num_windows * Batch
        n_levels is always one. Multi-label not supported (TODO)
        :param query                       (B_, Length_{query}, C)
        :param reference_points            (1, Length_{query}, n_levels, 3),
                                           range in [0, 1], top-left (0, 0, 0),
                                           bottom-right (1, 1, 1),
                                           including padding area
                                        or (N, Length_{query}, n_levels, 6),
                                           add additional (d, w, h) to form
                                           reference 3D boxes
        :param input_flatten               (B_, DxHxW, C)
        :param input_spatial_shapes        (n_levels, 2),
                                           [(H_0, W_0), (H_1, W_1),
                                            ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ),
                                           [
                                            0,
                                            H_0*W_0, H_0*W_0+H_1*W_1,
                                            H_0*W_0+H_1*W_1+H_2*W_2,
                                            ...,
                                            H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}
                                           ]
        :param mask                        (nW, DxHxW, D, H, W),
                                           True for adjacent elements,
                                           False for non-adjacent elements

        :return output                     (N, Length_{query}, C)
        """
        B_, Len_q, _ = query.shape
        B_, Len_in, _ = input_flatten.shape
        nW, S, D, H, W = mask.shape
        B = B_ // nW
        assert (input_spatial_shapes[0]
                * input_spatial_shapes[1]
                * input_spatial_shapes[2]) == Len_in

        value = self.value_proj(input_flatten)

        value = value.view(
            B_, Len_in, self.n_heads, self.d_model // self.n_heads)
        query_reshape = self.query_map(query).view(B, nW, S, -1)
        query_reshape = torch.cat(
            [query_reshape, pos_embed[None].repeat(B, 1, 1, 1)],
            -1).view(B_, S, -1)
        sampling_offsets = self.sampling_offsets(query_reshape).view(
            B, nW, S, self.n_heads, self.n_levels, self.n_points, 3)
        attention_weights = self.attention_weights(query_reshape).view(
            B_, S, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(
            B_, S, self.n_heads, self.n_levels, self.n_points)

        # B_, Len_q, n_heads, n_levels, n_points, 3
        if reference_points.shape[-1] == 3:
            W_ = mask.cumsum(-1).flatten(2).max(-1)[0]
            H_ = mask.cumsum(-2).flatten(2).max(-1)[0]
            D_ = mask.cumsum(-3).flatten(2).max(-1)[0]
            normalizer = torch.stack([W_, H_, D_], -1)
            normalizer = normalizer.to(reference_points.device)
            sampling_locations = (
                reference_points[None, :, :, None, None, None, :]
                + sampling_offsets
                / normalizer[None, :, :, None, None, None, :]
            )

            sampling_locations = sampling_locations.view(
                B_, S, self.n_heads, self.n_levels, self.n_points, 3)
        elif reference_points.shape[-1] == 6:
            sampling_locations = (
                reference_points[:, :, None, :, None, :3]
                + sampling_offsets
                / self.n_points
                * reference_points[:, :, None, :, None, 3:]
                * 0.5
            )
        else:
            raise ValueError(
                'Last dim of reference_points must be 3 or 6, '
                'but got {} instead.'.format(reference_points.shape[-1])
            )
        
        im2col_step = B
        output = MS3DDeformAttnFunction.apply(
            value, torch.tensor([input_spatial_shapes]).to(value.device),
            torch.tensor((0, )).to(value.device),  # input_level_start_index
            sampling_locations, attention_weights, im2col_step)
        output = self.output_proj(output)
        return output
