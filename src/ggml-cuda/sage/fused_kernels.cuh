/*
 * Copyright (c) 2024 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "cp_async.cuh"
#include "numeric_conversion.cuh"
#include "reduction_utils.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace ggml_cuda_sage {

template <typename T>
__device__ __forceinline__ float convert_to_float(T val) {
    static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value,
                  "Only half and bfloat16 are supported");

    if constexpr (std::is_same<T, half>::value) {
        return __half2float(val);
    } else {
        return __bfloat162float(val);
    }
}

template <typename T>
__device__ __forceinline__ T convert_from_float(float val) {
    static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value,
                  "Only half and bfloat16 are supported");

    if constexpr (std::is_same<T, half>::value) {
        return __float2half_rn(val);
    } else {
        return __float2bfloat16_rn(val);
    }
}

template <uint32_t head_dim,
          uint32_t BLOCK_SIZE,
          uint32_t num_pack_per_thread = 1,
          bool has_sm_scale = false,
          bool sub_mean     = false,
          typename T>
__global__ void quant_int8_kernel(
        T * __restrict__ input,
        T * __restrict__ mean,
        int8_t * __restrict__ output,
        float * __restrict__ scale,
        float sm_scale,
        const uint32_t num_tokens,
        const uint32_t stride_bz_input,
        const uint32_t stride_seq_input,
        const uint32_t stride_h_input,
        const uint32_t stride_bz_mean,
        const uint32_t stride_h_mean,
        const uint32_t stride_bz_output,
        const uint32_t stride_seq_output,
        const uint32_t stride_h_output,
        const uint32_t stride_bz_scale,
        const uint32_t stride_h_scale) {
    static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value,
                  "Only half and bfloat16 are supported");
    static_assert(num_pack_per_thread > 0, "num_pack_per_thread must be > 0");

    constexpr uint32_t pack_size = 8; // float4 => 8 half/bfloat16
    constexpr uint32_t num_threads_per_token = head_dim / pack_size;
    static_assert(num_threads_per_token <= 32, "threads per token must be <= warp size");

    T x_val[num_pack_per_thread][8];
    T mean_val[8];
    float x_val_float[num_pack_per_thread][8];
    float mean_val_float[8];

    const uint32_t bx = blockIdx.x;
    const uint32_t head_id = blockIdx.y;
    const uint32_t batch_id = blockIdx.z;
    const uint32_t thread_id = threadIdx.x;

    const uint32_t thread_base_token = bx*BLOCK_SIZE + thread_id/num_threads_per_token;
    T * input_ptr_base = input + batch_id*stride_bz_input + head_id*stride_h_input
        + thread_base_token*stride_seq_input + (thread_id % num_threads_per_token)*pack_size;
    T * mean_ptr_base  = mean  + batch_id*stride_bz_mean  + head_id*stride_h_mean
        + (thread_id % num_threads_per_token)*pack_size;
    int8_t * output_ptr_base = output + batch_id*stride_bz_output + head_id*stride_h_output
        + thread_base_token*stride_seq_output + (thread_id % num_threads_per_token)*pack_size;
    float * scale_ptr_base   = scale  + batch_id*stride_bz_scale  + head_id*stride_h_scale + bx;

    if constexpr (sub_mean) {
        *(float4*)(&mean_val[0]) = *(float4*)(mean_ptr_base);
#pragma unroll
        for (uint32_t j = 0; j < 8; ++j) {
            mean_val_float[j] = convert_to_float(mean_val[j]);
        }
    }

    constexpr uint32_t iter_stride = BLOCK_SIZE / num_pack_per_thread;
    for (uint32_t i = 0; i < num_pack_per_thread; ++i) {
        if (thread_base_token + i*iter_stride < num_tokens) {
            *(float4*)(&x_val[i][0]) = *(float4*)(input_ptr_base + i*iter_stride*stride_seq_input);
#pragma unroll
            for (uint32_t j = 0; j < 8; ++j) {
                x_val_float[i][j] = convert_to_float(x_val[i][j]);
            }

            if constexpr (sub_mean) {
#pragma unroll
                for (uint32_t j = 0; j < 8; ++j) {
                    x_val_float[i][j] -= mean_val_float[j];
                }
            }

            if constexpr (has_sm_scale) {
#pragma unroll
                for (uint32_t j = 0; j < 8; ++j) {
                    x_val_float[i][j] *= sm_scale;
                }
            }
        } else {
#pragma unroll
            for (uint32_t j = 0; j < 8; ++j) {
                x_val_float[i][j] = 0.0f;
            }
        }
    }

    float amax_val = 1e-7f;
#pragma unroll
    for (uint32_t i = 0; i < num_pack_per_thread; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < 8; ++j) {
            amax_val = fmaxf(amax_val, fabsf(x_val_float[i][j]));
        }
    }

    __shared__ float s_amax;
    const float block_amax_val = vllm::blockReduceMax(amax_val);
    if (thread_id == 0) {
        s_amax = block_amax_val;
        scale_ptr_base[0] = s_amax / 127.0f;
    }
    __syncthreads();

    const float tmp_scale = 127.0f / s_amax;
    char4 o_val[num_pack_per_thread][2];
#pragma unroll
    for (uint32_t i = 0; i < num_pack_per_thread; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
            o_val[i][j] = make_char4(
                float_to_int8_rn(x_val_float[i][j*4 + 0] * tmp_scale),
                float_to_int8_rn(x_val_float[i][j*4 + 1] * tmp_scale),
                float_to_int8_rn(x_val_float[i][j*4 + 2] * tmp_scale),
                float_to_int8_rn(x_val_float[i][j*4 + 3] * tmp_scale));
        }
    }

#pragma unroll
    for (uint32_t i = 0; i < num_pack_per_thread; ++i) {
        if (thread_base_token + i*iter_stride < num_tokens) {
            *reinterpret_cast<float2*>(output_ptr_base + i*iter_stride*stride_seq_output) =
                *reinterpret_cast<float2*>(&o_val[i][0]);
        }
    }
}

template <uint32_t head_dim,
          uint32_t BLOCK_SIZE,
          uint32_t num_pack_per_thread = 1,
          typename T>
__global__ void sub_mean_kernel(
        T * __restrict__ input,
        T * __restrict__ mean,
        half * __restrict__ output,
        const uint32_t num_tokens,
        const uint32_t stride_bz_input,
        const uint32_t stride_seq_input,
        const uint32_t stride_h_input,
        const uint32_t stride_bz_mean,
        const uint32_t stride_h_mean,
        const uint32_t stride_bz_output,
        const uint32_t stride_seq_output,
        const uint32_t stride_h_output) {
    static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value,
                  "Only half and bfloat16 are supported");
    static_assert(num_pack_per_thread > 0, "num_pack_per_thread must be > 0");

    using T2 = typename std::conditional<std::is_same<T, half>::value, half2, nv_bfloat162>::type;

    constexpr uint32_t pack_size = 8;
    constexpr uint32_t num_threads_per_token = head_dim / pack_size;
    static_assert(num_threads_per_token <= 32, "threads per token must be <= warp size");

    T2 x_val[num_pack_per_thread][4];
    T2 mean_val[4];

    const uint32_t bx = blockIdx.x;
    const uint32_t head_id = blockIdx.y;
    const uint32_t batch_id = blockIdx.z;
    const uint32_t thread_id = threadIdx.x;

    const uint32_t thread_base_token = bx*BLOCK_SIZE + thread_id/num_threads_per_token;
    T * input_ptr_base = input + batch_id*stride_bz_input + head_id*stride_h_input
        + thread_base_token*stride_seq_input + (thread_id % num_threads_per_token)*pack_size;
    T * mean_ptr_base  = mean  + batch_id*stride_bz_mean  + head_id*stride_h_mean
        + (thread_id % num_threads_per_token)*pack_size;
    half * output_ptr_base = output + batch_id*stride_bz_output + head_id*stride_h_output
        + thread_base_token*stride_seq_output + (thread_id % num_threads_per_token)*pack_size;

    *(float4*)(&mean_val[0]) = *(float4*)(mean_ptr_base);

    constexpr uint32_t iter_stride = BLOCK_SIZE / num_pack_per_thread;
    for (uint32_t i = 0; i < num_pack_per_thread; ++i) {
        if (thread_base_token + i*iter_stride < num_tokens) {
            *(float4*)(&x_val[i][0]) = *(float4*)(input_ptr_base + i*iter_stride*stride_seq_input);
#pragma unroll
            for (uint32_t j = 0; j < 4; ++j) {
                x_val[i][j] = __hsub2(x_val[i][j], mean_val[j]);
                if constexpr (std::is_same<T, nv_bfloat16>::value) {
                    ((half2*)x_val[i])[j] = __float22half2_rn(__bfloat1622float2(x_val[i][j]));
                }
            }
        }
    }

#pragma unroll
    for (uint32_t i = 0; i < num_pack_per_thread; ++i) {
        if (thread_base_token + i*iter_stride < num_tokens) {
            *reinterpret_cast<float4*>(output_ptr_base + i*iter_stride*stride_seq_output) =
                *reinterpret_cast<float4*>(&x_val[i][0]);
        }
    }
}

template <uint32_t head_dim,
          uint32_t CTA_SIZE,
          bool pad_zero = false,
          typename T>
__global__ void transpose_pad_permute_kernel(
        T * __restrict__ input,
        T * __restrict__ output,
        const uint32_t num_tokens,
        const uint32_t stride_bz_input,
        const uint32_t stride_seq_input,
        const uint32_t stride_h_input,
        const uint32_t stride_bz_output,
        const uint32_t stride_d_output,
        const uint32_t stride_h_output) {
    static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value,
                  "Only half and bfloat16 are supported");

    constexpr uint32_t pack_size = 8;
    const uint32_t num_threads_per_token = head_dim / pack_size;
    const uint32_t num_threads_per_cta = CTA_SIZE / pack_size;

    const uint32_t bx = blockIdx.x;
    const uint32_t head_id = blockIdx.y;
    const uint32_t batch_id = blockIdx.z;
    const uint32_t thread_id = threadIdx.x;

    const uint32_t thread_base_token = bx*CTA_SIZE + thread_id/num_threads_per_token;

    T * input_ptr_base = input + batch_id*stride_bz_input + head_id*stride_h_input
        + thread_base_token*stride_seq_input + (thread_id % num_threads_per_token)*pack_size;
    T * output_ptr_base = output + batch_id*stride_bz_output + head_id*stride_h_output
        + bx*CTA_SIZE + (thread_id % num_threads_per_cta)*pack_size
        + (thread_id/num_threads_per_cta)*stride_d_output;

    __shared__ T shared_load[CTA_SIZE][head_dim];
    __shared__ T shared_store[head_dim][CTA_SIZE];

    const uint32_t smem_load_row_base = ((thread_id/num_threads_per_token)/16)*16;
    const uint32_t smem_load_row_mod = (thread_id/num_threads_per_token) % 16;
    const uint32_t smem_load_row = smem_load_row_base
        + (smem_load_row_mod/8)*2
        + ((smem_load_row_mod/2) % 4)*4
        + (smem_load_row_mod % 2);

    constexpr cp_async::SharedMemFillMode fill_mode =
        pad_zero ? cp_async::SharedMemFillMode::kFillZero : cp_async::SharedMemFillMode::kNoFill;

    cp_async::pred_load_128b<cp_async::PrefetchMode::kNoPrefetch, fill_mode>(
        shared_load[smem_load_row] + (thread_id % num_threads_per_token)*pack_size,
        input_ptr_base,
        thread_base_token < num_tokens);
    cp_async::commit_group();
    cp_async::wait_group<0>();
    __syncthreads();

    const uint32_t smem_row_base = thread_id % CTA_SIZE;
    const uint32_t smem_col_base = thread_id / CTA_SIZE;
    const uint32_t smem_col_stride = head_dim / 8;

#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
        shared_store[smem_col_base + i*smem_col_stride][smem_row_base] =
            shared_load[smem_row_base][smem_col_base + i*smem_col_stride];
    }
    __syncthreads();

    *(float4*)(output_ptr_base) =
        *(float4*)(&shared_store[thread_id/num_threads_per_cta][(thread_id % num_threads_per_cta)*pack_size]);
}

template <uint32_t pad_size,
          bool sub_mean = false,
          typename T>
__global__ void mean_scale_kernel(
        T * __restrict__ input,
        int8_t * __restrict__ output,
        float * __restrict__ mean,
        float * __restrict__ scale,
        const float scale_max,
        const uint32_t num_tokens,
        const uint32_t stride_bz_input,
        const uint32_t stride_d_input,
        const uint32_t stride_h_input,
        const uint32_t stride_bz_output,
        const uint32_t stride_d_output,
        const uint32_t stride_h_output,
        const uint32_t stride_bz_mean,
        const uint32_t stride_h_mean,
        const uint32_t stride_bz_scale,
        const uint32_t stride_h_scale) {
    static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value,
                  "Only half and bfloat16 are supported");

    constexpr uint32_t pack_size = 8;

    const uint32_t head_id = blockIdx.x;
    const uint32_t batch_id = blockIdx.y;
    const uint32_t d_id = blockIdx.z;
    const uint32_t thread_id = threadIdx.x;

    const uint32_t num_threads = blockDim.x;
    const uint32_t gmem_stride = num_threads*pack_size;
    const uint32_t fp8_padded_num_tokens = ((num_tokens + 15) / 16) * 16;
    uint32_t num_iters = fp8_padded_num_tokens / gmem_stride
        + ((fp8_padded_num_tokens % gmem_stride) > thread_id*pack_size);

    T * input_ptr_base = input + batch_id*stride_bz_input + head_id*stride_h_input
        + d_id*stride_d_input + thread_id*pack_size;
    int8_t * output_ptr_base = output + batch_id*stride_bz_output + head_id*stride_h_output
        + d_id*stride_d_output + thread_id*pack_size;

    T x_val[8];
    float x_val_float[8];
    uint32_t x_val_fp8[2];

    float max_val = -1e6f;
    float min_val =  1e6f;
    float sum_val = 0.0f;

    for (uint32_t i = 0; i < num_iters; ++i) {
        *(float4*)(&x_val[0]) = *(float4*)(input_ptr_base + i*gmem_stride);
#pragma unroll
        for (uint32_t j = 0; j < 8; ++j) {
            const float x_tmp = convert_to_float(x_val[j]);
            max_val = fmaxf(max_val, x_tmp);
            min_val = fminf(min_val, x_tmp);
            if constexpr (sub_mean) {
                sum_val += x_tmp;
            }
        }
    }

    __shared__ float s_amax_val;
    __shared__ float s_mean_val;

    const float block_max_val = vllm::blockReduceMax(max_val);
    const float block_min_val = vllm::blockReduceMin(min_val);
    float block_sum_val = 0.0f;
    if constexpr (sub_mean) {
        block_sum_val = vllm::blockReduceSum(sum_val);
    }

    if (thread_id == 0) {
        s_mean_val = block_sum_val / fp8_padded_num_tokens;
        if constexpr (sub_mean) {
            s_amax_val = fmaxf(fabsf(block_max_val - s_mean_val),
                               fabsf(block_min_val - s_mean_val));
            mean[batch_id*stride_bz_mean + head_id*stride_h_mean + d_id] = s_mean_val;
        } else {
            s_amax_val = fmaxf(fabsf(block_max_val), fabsf(block_min_val));
        }
        scale[batch_id*stride_bz_scale + head_id*stride_h_scale + d_id] = s_amax_val / scale_max;
    }
    __syncthreads();

    const float mean_val = s_mean_val;
    const float recp_scale = scale_max / s_amax_val;

    const uint32_t padded_num_tokens = ((num_tokens + pad_size - 1) / pad_size) * pad_size;
    num_iters = padded_num_tokens / gmem_stride
        + ((padded_num_tokens % gmem_stride) > thread_id*pack_size);

    for (uint32_t i = 0; i < num_iters; ++i) {
        *(float4*)(&x_val[0]) = *(float4*)(input_ptr_base + i*gmem_stride);
#pragma unroll
        for (uint32_t j = 0; j < 8; ++j) {
            x_val_float[j] = convert_to_float(x_val[j]);
            if constexpr (sub_mean) {
                x_val_float[j] = (x_val_float[j] - mean_val) * recp_scale;
            } else {
                x_val_float[j] *= recp_scale;
            }
        }

        floatx4_to_e4m3x4(x_val_fp8, x_val_float, x_val_float + 2);
        floatx4_to_e4m3x4(x_val_fp8 + 1, x_val_float + 4, x_val_float + 6);

        *(uint2*)(output_ptr_base + i*gmem_stride) = *(uint2*)(&x_val_fp8[0]);
    }
}

} // namespace ggml_cuda_sage
