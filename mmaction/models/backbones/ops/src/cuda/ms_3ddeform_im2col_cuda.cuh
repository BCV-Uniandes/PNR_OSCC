/*!
**************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************
* Modified from DCN (https://github.com/msracver/Deformable-ConvNets)
* Copyright (c) 2018 Microsoft
**************************************************************************
*/

#include <cstdio>
#include <algorithm>
#include <cstring>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THCAtomics.cuh>

#define CUDA_KERNEL_LOOP(i, n)                          \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;   \
      i < (n);                                          \
      i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 512;
inline int GET_BLOCKS(const int N, const int num_threads)
{
  return (N + num_threads - 1) / num_threads;
}


// CHECKED!
template <typename scalar_t>
__device__ scalar_t ms_3ddeform_attn_im2col_trilinear(const scalar_t* &bottom_data,
                                                   const int &depth, const int &height, const int &width, const int &nheads, const int &channels,
                                                   const scalar_t &d, const scalar_t &h, const scalar_t &w, const int &m, const int &c)
{
  const int d_low = floor(d);
  const int h_low = floor(h);
  const int w_low = floor(w);
  const int d_high = d_low + 1;
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const scalar_t ld = d - d_low;
  const scalar_t lh = h - h_low;
  const scalar_t lw = w - w_low;
  const scalar_t hd = 1 - ld, hh = 1 - lh, hw = 1 - lw;

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int d_stride = height * h_stride;
  const int d_low_ptr_offset = d_low * d_stride;
  const int d_high_ptr_offset = d_low_ptr_offset + d_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c;

  scalar_t v1 = 0;
  if (d_low >= 0 && h_low >= 0 && w_low >= 0)
  {
    const int ptr1 = d_low_ptr_offset + h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
  }
  scalar_t v2 = 0;
  if (d_low >= 0 && h_low >= 0 && w_high <= width - 1)
  {
    const int ptr2 = d_low_ptr_offset + h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
  }
  scalar_t v3 = 0;
  if (d_low >= 0 && h_high <= height - 1 && w_low >= 0)
  {
    const int ptr3 = d_low_ptr_offset + h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
  }
  scalar_t v4 = 0;
  if (d_low >= 0 && h_high <= height - 1 && w_high <= width - 1)
  {
    const int ptr4 = d_low_ptr_offset + h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
  }
  scalar_t v5 = 0;
  if (d_high <= depth - 1 && h_low >= 0 && w_low >= 0)
  {
    const int ptr5 = d_high_ptr_offset + h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v5 = bottom_data[ptr5];
  }
  scalar_t v6 = 0;
  if (d_high <= depth - 1 && h_low >= 0 && w_high <= width - 1)
  {
    const int ptr6 = d_high_ptr_offset + h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v6 = bottom_data[ptr6];
  }
  scalar_t v7 = 0;
  if (d_high <= depth - 1 && h_high <= height - 1 && w_low >= 0)
  {
    const int ptr7 = d_high_ptr_offset + h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v7 = bottom_data[ptr7];
  }
  scalar_t v8 = 0;
  if (d_high <= depth - 1 && h_high <= height - 1 && w_high <= width - 1)
  {
    const int ptr8 = d_high_ptr_offset + h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v8 = bottom_data[ptr8];
  }

  const scalar_t w1 = hd * hh * hw, w2 = hd * hh * lw, w3 = hd * lh * hw, w4 = hd * lh * lw;
  const scalar_t w5 = ld * hh * hw, w6 = ld * hh * lw, w7 = ld * lh * hw, w8 = ld * lh * lw;

  const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4 + w5 * v5 + w6 * v6 + w7 * v7 + w8 * v8);
  return val;
}


// CHECKED!
template <typename scalar_t>
__device__ void ms_3ddeform_attn_col2im_trilinear(const scalar_t* &bottom_data,
                                                   const int &depth, const int &height, const int &width, const int &nheads, const int &channels,
                                                   const scalar_t &d, const scalar_t &h, const scalar_t &w, const int &m, const int &c,
                                                   const scalar_t &top_grad,
                                                   const scalar_t &attn_weight,
                                                   scalar_t* &grad_value, 
                                                   scalar_t* grad_sampling_loc,
                                                   scalar_t* grad_attn_weight)
{
  const int d_low = floor(d);
  const int h_low = floor(h);
  const int w_low = floor(w);
  const int d_high = d_low + 1;
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const scalar_t ld = d - d_low;
  const scalar_t lh = h - h_low;
  const scalar_t lw = w - w_low;
  const scalar_t hd = 1 - ld, hh = 1 - lh, hw = 1 - lw;

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int d_stride = height * h_stride;
  const int d_low_ptr_offset = d_low * d_stride;
  const int d_high_ptr_offset = d_low_ptr_offset + d_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c;

  const scalar_t w1 = hd * hh * hw, w2 = hd * hh * lw, w3 = hd * lh * hw, w4 = hd * lh * lw;
  const scalar_t w5 = ld * hh * hw, w6 = ld * hh * lw, w7 = ld * lh * hw, w8 = ld * lh * lw;
  const scalar_t top_grad_value = top_grad * attn_weight;
  scalar_t grad_d_weight = 0, grad_h_weight = 0, grad_w_weight = 0;

  scalar_t v1 = 0;
  if (d_low >= 0 && h_low >= 0 && w_low >= 0)
  {
    const int ptr1 = d_low_ptr_offset + h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
    grad_d_weight -= hw * hh * v1;
    grad_h_weight -= hd * hw * v1;
    grad_w_weight -= hd * hh * v1;
    atomicAdd(grad_value + ptr1, w1 * top_grad_value);
  }
  scalar_t v2 = 0;
  if (d_low >= 0 && h_low >= 0 && w_high <= width - 1)
  {
    const int ptr2 = d_low_ptr_offset + h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
    grad_d_weight -= lw * hh * v2;
    grad_h_weight -= hd * lw * v2;
    grad_w_weight += hd * hh * v2;
    atomicAdd(grad_value + ptr2, w2 * top_grad_value);
  }
  scalar_t v3 = 0;
  if (d_low >= 0 && h_high <= height - 1 && w_low >= 0)
  {
    const int ptr3 = d_low_ptr_offset + h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
    grad_d_weight -= hw * lh * v3;
    grad_h_weight += hd * hw * v3;
    grad_w_weight -= hd * lh * v3;
    atomicAdd(grad_value + ptr3, w3 * top_grad_value); 
  }
  scalar_t v4 = 0;
  if (d_low >= 0 && h_high <= height - 1 && w_high <= width - 1)
  {
    const int ptr4 = d_low_ptr_offset + h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
    grad_d_weight -= lw * lh * v4;
    grad_h_weight += hd * lw * v4;
    grad_w_weight += hd * lh * v4;
    atomicAdd(grad_value + ptr4, w4 * top_grad_value);
  }
  scalar_t v5 = 0;
  if (d_high <= depth - 1 && h_low >= 0 && w_low >= 0)
  {
    const int ptr5 = d_high_ptr_offset + h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v5 = bottom_data[ptr5];
    grad_d_weight += hw * hh * v5;
    grad_h_weight -= ld * hw * v5;
    grad_w_weight -= ld * hh * v5;
    atomicAdd(grad_value + ptr5, w5 * top_grad_value);
  }
  scalar_t v6 = 0;
  if (d_high <= depth - 1 && h_low >= 0 && w_high <= width - 1)
  {
    const int ptr6 = d_high_ptr_offset + h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v6 = bottom_data[ptr6];
    grad_d_weight += lw * hh * v6;
    grad_h_weight -= ld * lw * v6;
    grad_w_weight += ld * hh * v6;
    atomicAdd(grad_value + ptr6, w6 * top_grad_value);
  }
  scalar_t v7 = 0;
  if (d_high <= depth - 1 && h_high <= height - 1 && w_low >= 0)
  {
    const int ptr7 = d_high_ptr_offset + h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v7 = bottom_data[ptr7];
    grad_d_weight += hw * lh * v7;
    grad_h_weight += ld * hw * v7;
    grad_w_weight -= ld * lh * v7;
    atomicAdd(grad_value + ptr7, w7 * top_grad_value); 
  }
  scalar_t v8 = 0;
  if (d_high <= depth - 1 && h_high <= height - 1 && w_high <= width - 1)
  {
    const int ptr8 = d_high_ptr_offset + h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v8 = bottom_data[ptr8];
    grad_d_weight += lw * lh * v8;
    grad_h_weight += ld * lw * v8;
    grad_w_weight += ld * lh * v8;
    atomicAdd(grad_value + ptr8, w8 * top_grad_value);
  }

  const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4 + w5 * v5 + w6 * v6 + w7 * v7 + w8 * v8);
  *grad_attn_weight = top_grad * val;
  *grad_sampling_loc = width * grad_w_weight * top_grad_value;
  *(grad_sampling_loc + 1) = height * grad_h_weight * top_grad_value;
  *(grad_sampling_loc + 2) = depth * grad_d_weight * top_grad_value;
}


template <typename scalar_t>
__device__ void ms_3ddeform_attn_col2im_trilinear_gm(const scalar_t* &bottom_data, 
                                                   const int &depth, const int &height, const int &width, const int &nheads, const int &channels,
                                                   const scalar_t &d, const scalar_t &h, const scalar_t &w, const int &m, const int &c,
                                                   const scalar_t &top_grad,
                                                   const scalar_t &attn_weight,
                                                   scalar_t* &grad_value, 
                                                   scalar_t* grad_sampling_loc,
                                                   scalar_t* grad_attn_weight)
{
  const int d_low = floor(d);
  const int h_low = floor(h);
  const int w_low = floor(w);
  const int d_high = d_low + 1;
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const scalar_t ld = d - d_low;
  const scalar_t lh = h - h_low;
  const scalar_t lw = w - w_low;
  const scalar_t hd = 1 - ld, hh = 1 - lh, hw = 1 - lw;

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int d_stride = height * h_stride;
  const int d_low_ptr_offset = d_low * d_stride;
  const int d_high_ptr_offset = d_low_ptr_offset + d_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c;

  const scalar_t w1 = hd * hh * hw, w2 = hd * hh * lw, w3 = hd * lh * hw, w4 = hd * lh * lw;
  const scalar_t w5 = ld * hh * hw, w6 = ld * hh * lw, w7 = ld * lh * hw, w8 = ld * lh * lw;
  const scalar_t top_grad_value = top_grad * attn_weight;
  scalar_t grad_d_weight = 0, grad_h_weight = 0, grad_w_weight = 0;

  scalar_t v1 = 0;
  if (d_low >= 0 && h_low >= 0 && w_low >= 0)
  {
    const int ptr1 = d_low_ptr_offset + h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
    grad_d_weight -= hw * hh * v1;
    grad_h_weight -= hd * hw * v1;
    grad_w_weight -= hd * hh * v1;
    atomicAdd(grad_value + ptr1, w1 * top_grad_value);
  }
  scalar_t v2 = 0;
  if (d_low >= 0 && h_low >= 0 && w_high <= width - 1)
  {
    const int ptr2 = d_low_ptr_offset + h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
    grad_d_weight -= lw * hh * v2;
    grad_h_weight -= hd * lw * v2;
    grad_w_weight += hd * hh * v2;
    atomicAdd(grad_value + ptr2, w2 * top_grad_value);
  }
  scalar_t v3 = 0;
  if (d_low >= 0 && h_high <= height - 1 && w_low >= 0)
  {
    const int ptr3 = d_low_ptr_offset + h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
    grad_d_weight -= hw * lh * v3;
    grad_h_weight += hd * hw * v3;
    grad_w_weight -= hd * lh * v3;
    atomicAdd(grad_value + ptr3, w3 * top_grad_value); 
  }
  scalar_t v4 = 0;
  if (d_low >= 0 && h_high <= height - 1 && w_high <= width - 1)
  {
    const int ptr4 = d_low_ptr_offset + h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
    grad_d_weight -= lw * lh * v4;
    grad_h_weight += hd * lw * v4;
    grad_w_weight += hd * lh * v4;
    atomicAdd(grad_value + ptr4, w4 * top_grad_value);
  }
  scalar_t v5 = 0;
  if (d_high <= depth - 1 && h_low >= 0 && w_low >= 0)
  {
    const int ptr5 = d_high_ptr_offset + h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v5 = bottom_data[ptr5];
    grad_d_weight += hw * hh * v5;
    grad_h_weight -= ld * hw * v5;
    grad_w_weight -= ld * hh * v5;
    atomicAdd(grad_value + ptr5, w5 * top_grad_value);
  }
  scalar_t v6 = 0;
  if (d_high <= depth - 1 && h_low >= 0 && w_high <= width - 1)
  {
    const int ptr6 = d_high_ptr_offset + h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v6 = bottom_data[ptr6];
    grad_d_weight += lw * hh * v6;
    grad_h_weight -= ld * lw * v6;
    grad_w_weight += ld * hh * v6;
    atomicAdd(grad_value + ptr6, w6 * top_grad_value);
  }
  scalar_t v7 = 0;
  if (d_high <= depth - 1 && h_high <= height - 1 && w_low >= 0)
  {
    const int ptr7 = d_high_ptr_offset + h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v7 = bottom_data[ptr7];
    grad_d_weight += hw * lh * v7;
    grad_h_weight += ld * hw * v7;
    grad_w_weight -= ld * lh * v7;
    atomicAdd(grad_value + ptr7, w7 * top_grad_value); 
  }
  scalar_t v8 = 0;
  if (d_high <= depth - 1 && h_high <= height - 1 && w_high <= width - 1)
  {
    const int ptr8 = d_high_ptr_offset + h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v8 = bottom_data[ptr8];
    grad_d_weight += lw * lh * v8;
    grad_h_weight += ld * lw * v8;
    grad_w_weight += ld * lh * v8;
    atomicAdd(grad_value + ptr8, w8 * top_grad_value);
  }

  const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4 + w5 * v5 + w6 * v6 + w7 * v7 + w8 * v8);
  atomicAdd(grad_attn_weight, top_grad * val); 
  atomicAdd(grad_sampling_loc, width * grad_w_weight * top_grad_value);
  atomicAdd(grad_sampling_loc + 1, height * grad_h_weight * top_grad_value);
  atomicAdd(grad_sampling_loc + 2, depth * grad_d_weight * top_grad_value);
}


// CHECKED!
template <typename scalar_t>
__global__ void ms_3ddeformable_im2col_gpu_kernel(const int n,
                                                const scalar_t *data_value, 
                                                const int64_t *data_spatial_shapes,
                                                const int64_t *data_level_start_index, 
                                                const scalar_t *data_sampling_loc,
                                                const scalar_t *data_attn_weight,
                                                const int batch_size, 
                                                const int spatial_size, 
                                                const int num_heads,
                                                const int channels, 
                                                const int num_levels,
                                                const int num_query,
                                                const int num_point,
                                                scalar_t *data_col)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp; 
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    scalar_t *data_col_ptr = data_col + index;
    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr * 3;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;
    scalar_t col = 0;

    for (int l_col=0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col * 3;
      const int spatial_d = data_spatial_shapes[spatial_h_ptr];
      const int spatial_h = data_spatial_shapes[spatial_h_ptr + 1];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 2];
      const scalar_t *data_value_ptr = data_value + (data_value_ptr_init_offset + level_start_id * qid_stride);
      for (int p_col=0; p_col < num_point; ++p_col)
      {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t loc_d = data_sampling_loc[data_loc_w_ptr + 2];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t d_im = loc_d * spatial_d - 0.5;
        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;

        if (h_im > -1 && w_im > -1 && d_im > -1 && h_im < spatial_h && w_im < spatial_w && d_im < spatial_d)
        {
          col += ms_3ddeform_attn_im2col_trilinear(data_value_ptr, spatial_d, spatial_h, spatial_w, num_heads, channels, d_im, h_im, w_im, m_col, c_col) * weight;
        }

        data_weight_ptr += 1;
        data_loc_w_ptr += 3;
      }
    }
    *data_col_ptr = col;
  }
}

// CHECKED!
template <typename scalar_t, unsigned int blockSize>
__global__ void ms_3ddeformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1(const int n,
                                                const scalar_t *grad_col,
                                                const scalar_t *data_value,
                                                const int64_t *data_spatial_shapes,
                                                const int64_t *data_level_start_index, 
                                                const scalar_t *data_sampling_loc,
                                                const scalar_t *data_attn_weight,
                                                const int batch_size, 
                                                const int spatial_size, 
                                                const int num_heads,
                                                const int channels, 
                                                const int num_levels,
                                                const int num_query,
                                                const int num_point,
                                                scalar_t *grad_value,
                                                scalar_t *grad_sampling_loc,
                                                scalar_t *grad_attn_weight)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    __shared__ scalar_t cache_grad_sampling_loc[blockSize * 3];  // LD
    __shared__ scalar_t cache_grad_attn_weight[blockSize];
    unsigned int tid = threadIdx.x;
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp; 
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr * 3;  // LD
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr * 3;  // LD
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 3;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col=0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col * 3;
      const int spatial_d = data_spatial_shapes[spatial_h_ptr];
      const int spatial_h = data_spatial_shapes[spatial_h_ptr + 1];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 2];
      const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col=0; p_col < num_point; ++p_col)
      {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t loc_d = data_sampling_loc[data_loc_w_ptr + 2];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t d_im = loc_d * spatial_d - 0.5;
        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc+(threadIdx.x * 3)) = 0;  // LD
        *(cache_grad_sampling_loc+((threadIdx.x * 3) + 1)) = 0;  // LD
        *(cache_grad_sampling_loc+((threadIdx.x * 3) + 2)) = 0;  // LD
        *(cache_grad_attn_weight+threadIdx.x)=0;
        if (h_im > -1 && w_im > -1 && d_im > -1 && h_im < spatial_h && w_im < spatial_w && d_im < spatial_d)
        {
          ms_3ddeform_attn_col2im_trilinear(
            data_value_ptr, spatial_d, spatial_h, spatial_w, num_heads, channels, d_im, h_im, w_im, m_col, c_col,
            top_grad, weight, grad_value_ptr, 
            cache_grad_sampling_loc+(threadIdx.x * 3), cache_grad_attn_weight+threadIdx.x);  // LD
        }
        
        __syncthreads();
        if (tid == 0)
        {
          scalar_t _grad_w=cache_grad_sampling_loc[0], _grad_h=cache_grad_sampling_loc[1], _grad_d=cache_grad_sampling_loc[2], _grad_a=cache_grad_attn_weight[0];
          int sid=3;
          for (unsigned int tid = 1; tid < blockSize; ++tid)
          {
            _grad_w += cache_grad_sampling_loc[sid];
            _grad_h += cache_grad_sampling_loc[sid + 1];
            _grad_d += cache_grad_sampling_loc[sid + 2];
            _grad_a += cache_grad_attn_weight[tid];
            sid += 3;
          }
          
          
          *grad_sampling_loc = _grad_w;
          *(grad_sampling_loc + 1) = _grad_h;
          *(grad_sampling_loc + 2) = _grad_d;
          *grad_attn_weight = _grad_a;
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 3;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}


// CHECKED!
template <typename scalar_t, unsigned int blockSize>
__global__ void ms_3ddeformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2(const int n,
                                                const scalar_t *grad_col,
                                                const scalar_t *data_value,
                                                const int64_t *data_spatial_shapes,
                                                const int64_t *data_level_start_index, 
                                                const scalar_t *data_sampling_loc,
                                                const scalar_t *data_attn_weight,
                                                const int batch_size, 
                                                const int spatial_size, 
                                                const int num_heads,
                                                const int channels, 
                                                const int num_levels,
                                                const int num_query,
                                                const int num_point,
                                                scalar_t *grad_value,
                                                scalar_t *grad_sampling_loc,
                                                scalar_t *grad_attn_weight)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    __shared__ scalar_t cache_grad_sampling_loc[blockSize * 3];  // LD
    __shared__ scalar_t cache_grad_attn_weight[blockSize];
    unsigned int tid = threadIdx.x;
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp; 
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr * 3;  // LD
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr * 3;  // LD
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 3;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col=0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col * 3;
      const int spatial_d = data_spatial_shapes[spatial_h_ptr];
      const int spatial_h = data_spatial_shapes[spatial_h_ptr + 1];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 2];
      const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col=0; p_col < num_point; ++p_col)
      {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t loc_d = data_sampling_loc[data_loc_w_ptr + 2];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t d_im = loc_d * spatial_d - 0.5;
        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc+(threadIdx.x * 3)) = 0;  // LD
        *(cache_grad_sampling_loc+((threadIdx.x * 3) + 1)) = 0;  // LD
        *(cache_grad_sampling_loc+((threadIdx.x * 3) + 2)) = 0;  // LD
        *(cache_grad_attn_weight+threadIdx.x)=0;
        if (h_im > -1 && w_im > -1 && d_im > -1 && h_im < spatial_h && w_im < spatial_w && d_im < spatial_d)
        {
          ms_3ddeform_attn_col2im_trilinear(
            data_value_ptr, spatial_d, spatial_h, spatial_w, num_heads, channels, d_im, h_im, w_im, m_col, c_col,
            top_grad, weight, grad_value_ptr, 
            cache_grad_sampling_loc+(threadIdx.x * 3), cache_grad_attn_weight+threadIdx.x);  // LD
        }
        
        __syncthreads();

        for (unsigned int s=blockSize/3; s>0; s/3)  // LD
        {
          if (tid < s) {
            const unsigned int xid1 = tid * 3;  // LD
            const unsigned int xid2 = (tid + s) * 3;  // LD
            cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + s];
            cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2];
            cache_grad_sampling_loc[xid1 + 1] += cache_grad_sampling_loc[xid2 + 1];
            cache_grad_sampling_loc[xid1 + 2] += cache_grad_sampling_loc[xid2 + 2];  // LD
          }
          __syncthreads();
        }

        if (tid == 0)
        { 
          *grad_sampling_loc = cache_grad_sampling_loc[0];
          *(grad_sampling_loc + 1) = cache_grad_sampling_loc[1];
          *(grad_sampling_loc + 2) = cache_grad_sampling_loc[2];
          *grad_attn_weight = cache_grad_attn_weight[0];
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 3;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}


// CHECKED! --> Until 8!
template <typename scalar_t>
__global__ void ms_3ddeformable_col2im_gpu_kernel_shm_reduce_v1(const int n,
                                                const scalar_t *grad_col,
                                                const scalar_t *data_value,
                                                const int64_t *data_spatial_shapes,
                                                const int64_t *data_level_start_index, 
                                                const scalar_t *data_sampling_loc,
                                                const scalar_t *data_attn_weight,
                                                const int batch_size, 
                                                const int spatial_size, 
                                                const int num_heads,
                                                const int channels, 
                                                const int num_levels,
                                                const int num_query,
                                                const int num_point,
                                                scalar_t *grad_value,
                                                scalar_t *grad_sampling_loc,
                                                scalar_t *grad_attn_weight)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    extern __shared__ int _s[];
    scalar_t* cache_grad_sampling_loc = (scalar_t*)_s;
    scalar_t* cache_grad_attn_weight = cache_grad_sampling_loc + 3 * blockDim.x;
    unsigned int tid = threadIdx.x;
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp; 
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr * 3;  // LD
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr * 3;  // LD
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 3;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col=0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col * 3;  // LD
      const int spatial_d = data_spatial_shapes[spatial_h_ptr];
      const int spatial_h = data_spatial_shapes[spatial_h_ptr + 1];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 2];
      const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col=0; p_col < num_point; ++p_col)
      {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t loc_d = data_sampling_loc[data_loc_w_ptr + 2];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t d_im = loc_d * spatial_d - 0.5;
        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc+(threadIdx.x * 3)) = 0;  // LD
        *(cache_grad_sampling_loc+((threadIdx.x * 3) + 1)) = 0;  // LD
        *(cache_grad_sampling_loc+((threadIdx.x * 3) + 2)) = 0;  // ADDED
        *(cache_grad_attn_weight+threadIdx.x)=0;
        if (h_im > -1 && w_im > -1 && d_im > -1 && h_im < spatial_h && w_im < spatial_w && d_im < spatial_d)
        {
          ms_3ddeform_attn_col2im_trilinear(
            data_value_ptr, spatial_d, spatial_h, spatial_w, num_heads, channels, d_im, h_im, w_im, m_col, c_col,
            top_grad, weight, grad_value_ptr, 
            cache_grad_sampling_loc+(threadIdx.x * 3), cache_grad_attn_weight+threadIdx.x);  // LD
        }

        __syncthreads();
        if (tid == 0)
        {
          scalar_t _grad_w=cache_grad_sampling_loc[0], _grad_h=cache_grad_sampling_loc[1], _grad_d=cache_grad_sampling_loc[2], _grad_a=cache_grad_attn_weight[0];
          int sid=3;
          for (unsigned int tid = 1; tid < blockDim.x; ++tid)
          {
            _grad_w += cache_grad_sampling_loc[sid];
            _grad_h += cache_grad_sampling_loc[sid + 1];
            _grad_d += cache_grad_sampling_loc[sid + 2];
            _grad_a += cache_grad_attn_weight[tid];
            sid += 3;
          }
          
          
          *grad_sampling_loc = _grad_w;
          *(grad_sampling_loc + 1) = _grad_h;
          *(grad_sampling_loc + 2) = _grad_d;
          *grad_attn_weight = _grad_a;
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 3;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}

// CHECKED! --> Until 8!
template <typename scalar_t>
__global__ void ms_3ddeformable_col2im_gpu_kernel_shm_reduce_v2(const int n,
                                                const scalar_t *grad_col,
                                                const scalar_t *data_value,
                                                const int64_t *data_spatial_shapes,
                                                const int64_t *data_level_start_index, 
                                                const scalar_t *data_sampling_loc,
                                                const scalar_t *data_attn_weight,
                                                const int batch_size, 
                                                const int spatial_size, 
                                                const int num_heads,
                                                const int channels, 
                                                const int num_levels,
                                                const int num_query,
                                                const int num_point,
                                                scalar_t *grad_value,
                                                scalar_t *grad_sampling_loc,
                                                scalar_t *grad_attn_weight)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    extern __shared__ int _s[];
    scalar_t* cache_grad_sampling_loc = (scalar_t*)_s;
    scalar_t* cache_grad_attn_weight = cache_grad_sampling_loc + 3 * blockDim.x;
    unsigned int tid = threadIdx.x;
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp; 
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr * 3;
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr * 3;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 3;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col=0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col * 3;
      const int spatial_d = data_spatial_shapes[spatial_h_ptr];
      const int spatial_h = data_spatial_shapes[spatial_h_ptr + 1];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 2];
      const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col=0; p_col < num_point; ++p_col)
      {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t loc_d = data_sampling_loc[data_loc_w_ptr + 2];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t d_im = loc_d * spatial_d - 0.5;
        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc+(threadIdx.x * 3)) = 0;  // LD
        *(cache_grad_sampling_loc+((threadIdx.x * 3) + 1)) = 0;  // LD
        *(cache_grad_sampling_loc+((threadIdx.x * 3) + 2)) = 0;  // LD
        *(cache_grad_attn_weight+threadIdx.x)=0;
        if (h_im > -1 && w_im > -1 && d_im > -1 && h_im < spatial_h && w_im < spatial_w && d_im < spatial_d)
        {
          ms_3ddeform_attn_col2im_trilinear(
            data_value_ptr, spatial_d, spatial_h, spatial_w, num_heads, channels, d_im, h_im, w_im, m_col, c_col,
            top_grad, weight, grad_value_ptr, 
            cache_grad_sampling_loc+(threadIdx.x * 3), cache_grad_attn_weight+threadIdx.x);  // LD
        }
        
        __syncthreads();

        for (unsigned int s=blockDim.x/2, spre=blockDim.x; s>0; s/3, spre/3)  // LD
        {
          if (tid < s) {
            const unsigned int xid1 = tid * 3;  // LD
            const unsigned int xid2 = (tid + s) * 3;  // LD
            cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + s];
            cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2];
            cache_grad_sampling_loc[xid1 + 1] += cache_grad_sampling_loc[xid2 + 1];
            cache_grad_sampling_loc[xid1 + 2] += cache_grad_sampling_loc[xid2 + 2];
            if (tid + (s * 3) < spre)  // LD
            {
              cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + (s * 3)];   // LD
              cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2 + (s * 3)];  // LD
              cache_grad_sampling_loc[xid1 + 1] += cache_grad_sampling_loc[xid2 + 1 + (s * 3)];  // LD
              cache_grad_sampling_loc[xid1 + 2] += cache_grad_sampling_loc[xid2 + 2 + (s * 3)];  // LD
            } 
          }
          __syncthreads();
        }

        if (tid == 0)
        {
          *grad_sampling_loc = cache_grad_sampling_loc[0];
          *(grad_sampling_loc + 1) = cache_grad_sampling_loc[1];
          *(grad_sampling_loc + 2) = cache_grad_sampling_loc[2];
          *grad_attn_weight = cache_grad_attn_weight[0];
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 3;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}

template <typename scalar_t>
__global__ void ms_3ddeformable_col2im_gpu_kernel_shm_reduce_v2_multi_blocks(const int n,
                                                const scalar_t *grad_col,
                                                const scalar_t *data_value,
                                                const int64_t *data_spatial_shapes,
                                                const int64_t *data_level_start_index, 
                                                const scalar_t *data_sampling_loc,
                                                const scalar_t *data_attn_weight,
                                                const int batch_size, 
                                                const int spatial_size, 
                                                const int num_heads,
                                                const int channels, 
                                                const int num_levels,
                                                const int num_query,
                                                const int num_point,
                                                scalar_t *grad_value,
                                                scalar_t *grad_sampling_loc,
                                                scalar_t *grad_attn_weight)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    extern __shared__ int _s[];
    scalar_t* cache_grad_sampling_loc = (scalar_t*)_s;
    scalar_t* cache_grad_attn_weight = cache_grad_sampling_loc + 3 * blockDim.x;
    unsigned int tid = threadIdx.x;
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp; 
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr * 3;  // LD
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr * 3;  // LD
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 3;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col=0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col * 3;
      const int spatial_d = data_spatial_shapes[spatial_h_ptr];
      const int spatial_h = data_spatial_shapes[spatial_h_ptr + 1];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 2];
      const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col=0; p_col < num_point; ++p_col)
      {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t loc_d = data_sampling_loc[data_loc_w_ptr + 2];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t d_im = loc_d * spatial_d - 0.5;
        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc+(threadIdx.x * 3)) = 0;  // LD
        *(cache_grad_sampling_loc+((threadIdx.x * 3) + 1)) = 0;  // LD
        *(cache_grad_sampling_loc+((threadIdx.x * 3) + 2)) = 0;  // LD
        *(cache_grad_attn_weight+threadIdx.x)=0;
        if (h_im > -1 && w_im > -1 && d_im > -1 && h_im < spatial_h && w_im < spatial_w && d_im < spatial_d)
        {
          ms_3ddeform_attn_col2im_trilinear(
            data_value_ptr, spatial_d, spatial_h, spatial_w, num_heads, channels, d_im, h_im, w_im, m_col, c_col,
            top_grad, weight, grad_value_ptr, 
            cache_grad_sampling_loc+(threadIdx.x * 3), cache_grad_attn_weight+threadIdx.x);  // LD
        }
        
        __syncthreads();

        for (unsigned int s=blockDim.x/2, spre=blockDim.x; s>0; s/3, spre/3)  // LD
        {
          if (tid < s) {
            const unsigned int xid1 = tid * 3;  // LD
            const unsigned int xid2 = (tid + s) * 3;  // LD
            cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + s];
            cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2];
            cache_grad_sampling_loc[xid1 + 1] += cache_grad_sampling_loc[xid2 + 1];
            cache_grad_sampling_loc[xid1 + 2] += cache_grad_sampling_loc[xid2 + 2];
            if (tid + (s * 3) < spre)  // LD
            {
              cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + (s * 3)];  // LD
              cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2 + (s * 3)];  // LD
              cache_grad_sampling_loc[xid1 + 1] += cache_grad_sampling_loc[xid2 + 1 + (s * 3)];  // LD
              cache_grad_sampling_loc[xid1 + 2] += cache_grad_sampling_loc[xid2 + 2 + (s * 3)];  // LD
            }
          }
          __syncthreads();
        }

        if (tid == 0)
        {
          atomicAdd(grad_sampling_loc, cache_grad_sampling_loc[0]);
          atomicAdd(grad_sampling_loc + 1, cache_grad_sampling_loc[1]);
          atomicAdd(grad_sampling_loc + 2, cache_grad_sampling_loc[2]);
          atomicAdd(grad_attn_weight, cache_grad_attn_weight[0]);
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 3;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}


template <typename scalar_t>
__global__ void ms_3ddeformable_col2im_gpu_kernel_gm(const int n,
                                                const scalar_t *grad_col,
                                                const scalar_t *data_value,
                                                const int64_t *data_spatial_shapes,
                                                const int64_t *data_level_start_index, 
                                                const scalar_t *data_sampling_loc,
                                                const scalar_t *data_attn_weight,
                                                const int batch_size, 
                                                const int spatial_size, 
                                                const int num_heads,
                                                const int channels, 
                                                const int num_levels,
                                                const int num_query,
                                                const int num_point,
                                                scalar_t *grad_value,
                                                scalar_t *grad_sampling_loc,
                                                scalar_t *grad_attn_weight)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp; 
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr * 3;  // LD
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr * 3;  // LD
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 3;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col=0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col * 3;
      const int spatial_d = data_spatial_shapes[spatial_h_ptr];
      const int spatial_h = data_spatial_shapes[spatial_h_ptr + 1];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 2];
      const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col=0; p_col < num_point; ++p_col)
      {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t loc_d = data_sampling_loc[data_loc_w_ptr + 2];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t d_im = loc_d * spatial_d - 0.5;
        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        if (h_im > -1 && w_im > -1 && d_im > -1 && h_im < spatial_h && w_im < spatial_w && d_im < spatial_d)
        {
          ms_3ddeform_attn_col2im_trilinear_gm(
            data_value_ptr, spatial_d, spatial_h, spatial_w, num_heads, channels, d_im, h_im, w_im, m_col, c_col,
            top_grad, weight, grad_value_ptr, 
            grad_sampling_loc, grad_attn_weight);
        }
        data_weight_ptr += 1;
        data_loc_w_ptr += 3;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}


template <typename scalar_t>
void ms_3ddeformable_im2col_cuda(cudaStream_t stream,
                              const scalar_t* data_value,
                              const int64_t* data_spatial_shapes, 
                              const int64_t* data_level_start_index, 
                              const scalar_t* data_sampling_loc,
                              const scalar_t* data_attn_weight,
                              const int batch_size,
                              const int spatial_size, 
                              const int num_heads, 
                              const int channels, 
                              const int num_levels, 
                              const int num_query,
                              const int num_point,
                              scalar_t* data_col)
{
  const int num_kernels = batch_size * num_query * num_heads * channels;
  const int num_actual_kernels = batch_size * num_query * num_heads * channels;
  const int num_threads = CUDA_NUM_THREADS;
  ms_3ddeformable_im2col_gpu_kernel<scalar_t>
      <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
          0, stream>>>(
      num_kernels, data_value, data_spatial_shapes, data_level_start_index, data_sampling_loc, data_attn_weight, 
      batch_size, spatial_size, num_heads, channels, num_levels, num_query, num_point, data_col);
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in ms_3ddeformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }

}

template <typename scalar_t>
void ms_3ddeformable_col2im_cuda(cudaStream_t stream,
                              const scalar_t* grad_col,
                              const scalar_t* data_value,
                              const int64_t * data_spatial_shapes,
                              const int64_t * data_level_start_index,
                              const scalar_t * data_sampling_loc,
                              const scalar_t * data_attn_weight,
                              const int batch_size, 
                              const int spatial_size, 
                              const int num_heads,
                              const int channels, 
                              const int num_levels,
                              const int num_query,
                              const int num_point, 
                              scalar_t* grad_value,
                              scalar_t* grad_sampling_loc,
                              scalar_t* grad_attn_weight)
{
  const int num_threads = (channels > CUDA_NUM_THREADS)?CUDA_NUM_THREADS:channels;
  const int num_kernels = batch_size * num_query * num_heads * channels;
  const int num_actual_kernels = batch_size * num_query * num_heads * channels;
  if (channels > 1024)
  {
    if ((channels & 1023) == 0)
    {
      ms_3ddeformable_col2im_gpu_kernel_shm_reduce_v2_multi_blocks<scalar_t>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
              num_threads*3*sizeof(scalar_t), stream>>>(
                        num_kernels, 
                        grad_col,
                        data_value,
                        data_spatial_shapes,
                        data_level_start_index, 
                        data_sampling_loc,
                        data_attn_weight,
                        batch_size, 
                        spatial_size, 
                        num_heads,
                        channels, 
                        num_levels,
                        num_query,
                        num_point,
                        grad_value,
                        grad_sampling_loc,
                        grad_attn_weight);
    }
    else
    {
      ms_3ddeformable_col2im_gpu_kernel_gm<scalar_t>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      channels, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
    }
  }
  else{
    switch(channels)
    {
      case 1:
        ms_3ddeformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 1>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      channels, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 2:
        ms_3ddeformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 2>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      channels, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 4:
        ms_3ddeformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 4>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      channels, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 8:
        ms_3ddeformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 8>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      channels, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 16:
        ms_3ddeformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 16>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      channels, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 32:
        ms_3ddeformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 32>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      channels, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 64:
        ms_3ddeformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 64>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      channels, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 128:
        ms_3ddeformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 128>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      channels, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 256:
        ms_3ddeformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 256>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      channels, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 512:
        ms_3ddeformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 512>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      channels, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 1024:
        ms_3ddeformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 1024>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      channels, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      default:
        if (channels < 64)
        {
          ms_3ddeformable_col2im_gpu_kernel_shm_reduce_v1<scalar_t>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
              num_threads*3*sizeof(scalar_t), stream>>>(
                        num_kernels, 
                        grad_col,
                        data_value,
                        data_spatial_shapes,
                        data_level_start_index, 
                        data_sampling_loc,
                        data_attn_weight,
                        batch_size, 
                        spatial_size, 
                        num_heads,
                        channels, 
                        num_levels,
                        num_query,
                        num_point,
                        grad_value,
                        grad_sampling_loc,
                        grad_attn_weight);
        }
        else
        {
          ms_3ddeformable_col2im_gpu_kernel_shm_reduce_v2<scalar_t>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
              num_threads*3*sizeof(scalar_t), stream>>>(
                        num_kernels, 
                        grad_col,
                        data_value,
                        data_spatial_shapes,
                        data_level_start_index, 
                        data_sampling_loc,
                        data_attn_weight,
                        batch_size, 
                        spatial_size, 
                        num_heads,
                        channels, 
                        num_levels,
                        num_query,
                        num_point,
                        grad_value,
                        grad_sampling_loc,
                        grad_attn_weight);
        }
    }
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in ms_3ddeformable_col2im_cuda: %s\n", cudaGetErrorString(err));
  }

}