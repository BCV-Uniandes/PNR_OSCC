/*!
**************************************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/

#include "ms_3ddeform_attn.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ms_3ddeform_attn_forward", &ms_3ddeform_attn_forward, "ms_3ddeform_attn_forward");
  m.def("ms_3ddeform_attn_backward", &ms_3ddeform_attn_backward, "ms_3ddeform_attn_backward");
}