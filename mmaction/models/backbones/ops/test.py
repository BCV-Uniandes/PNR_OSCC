# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import torch
import torch.nn as nn
from torch.autograd import gradcheck

from functions.ms_3ddeform_attn_func import (
    MS3DDeformAttnFunction,
    ms_3ddeform_attn_core_pytorch
)


N, M, D = 2432, 4, 128  # Batch, Heads, Hidden Dim
Lq, L, P = 392, 1, 4  # Queries length, Levels, Points
# shapes = torch.as_tensor([(3, 3, 3), (2, 2, 2)], dtype=torch.long).cuda()
shapes = torch.as_tensor([(2, 2, 1)], dtype=torch.long).cuda()
level_start_index = torch.cat(
    (shapes.new_zeros((1, )), shapes.prod(1).cumsum(0)[:-1])
)
S = sum([(T*H*W).item() for T, H, W in shapes])  # Flattened inputs size

torch.manual_seed(8)


@torch.no_grad()
def check_forward_equal_with_pytorch_double():
    value = torch.rand(N, S, M, D).cuda() * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 3).cuda()
    attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
    attention_weights /= attention_weights.sum(
        -1, keepdim=True).sum(-2, keepdim=True)
    im2col_step = 3  # What does this do?
    output_pytorch = ms_3ddeform_attn_core_pytorch(
        value.double(), shapes, sampling_locations.double(),
        attention_weights.double()).detach().cpu()
    output_cuda = MS3DDeformAttnFunction.apply(
        value.double(), shapes, level_start_index, sampling_locations.double(),
        attention_weights.double(), im2col_step).detach().cpu()
    fwdok = torch.allclose(output_cuda, output_pytorch)
    max_abs_err = (output_cuda - output_pytorch).abs().max()
    max_rel_err = (
        (output_cuda - output_pytorch).abs() / output_pytorch.abs()
    ).max()

    print(f'* {fwdok} check_forward_equal_with_pytorch_double: ',
          f'max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}')


@torch.no_grad()
def check_forward_equal_with_pytorch_float():
    value = torch.rand(N, S, M, D).cuda() * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 3).cuda()
    attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
    attention_weights /= attention_weights.sum(
        -1, keepdim=True).sum(-2, keepdim=True)
    im2col_step = 3
    output_pytorch = ms_3ddeform_attn_core_pytorch(
        value, shapes, sampling_locations, attention_weights).detach().cpu()
    output_cuda = MS3DDeformAttnFunction.apply(
        value, shapes, level_start_index, sampling_locations,
        attention_weights, im2col_step).detach().cpu()
    fwdok = torch.allclose(output_cuda, output_pytorch, rtol=1e-2, atol=1e-3)
    max_abs_err = (output_cuda - output_pytorch).abs().max()
    max_rel_err = (
        (output_cuda - output_pytorch).abs() / output_pytorch.abs()
    ).max()

    print(f'* {fwdok} check_forward_equal_with_pytorch_float: ',
          f'max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}')


def check_gradient_numerical(
        channels=4, grad_value=True, grad_sampling_loc=True,
        grad_attn_weight=True):

    value = torch.rand(N, S, M, channels).cuda() * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 3).cuda()
    attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
    attention_weights /= attention_weights.sum(
        -1, keepdim=True).sum(-2, keepdim=True)
    im2col_step = 64
    func = MS3DDeformAttnFunction.apply

    value.requires_grad = grad_value
    sampling_locations.requires_grad = grad_sampling_loc
    attention_weights.requires_grad = grad_attn_weight

    gradok = gradcheck(
        func,
        (value.double(), shapes, level_start_index,
        sampling_locations.double(), attention_weights.double(),
        im2col_step),
        # raise_exception=False,
    )

    print(f'* {gradok} check_gradient_numerical(D={channels})')


if __name__ == '__main__':
    # check_forward_equal_with_pytorch_double()
    # check_forward_equal_with_pytorch_float()

    # for channels in [30, 32, 64, 71, 1025, 2048, 3096]:
    for channels in [8]:
        check_gradient_numerical(channels, True, True, True)
