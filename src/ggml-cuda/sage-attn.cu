#include "sage-attn.cuh"

#include "common.cuh"
#include "ggml-cuda.h"

#include "sage/fused_kernels.cuh"
#include "sage/qattn/qk_int_sv_f8_cuda_sm89.cuh"
#include "sage/qattn/attn_utils.cuh"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>

namespace {

template<typename T>
using cuda_type = T;

template<typename T>
constexpr ggml_type ggml_type_for();

template<>
constexpr ggml_type ggml_type_for<half>() {
    return GGML_TYPE_F16;
}

template<>
constexpr ggml_type ggml_type_for<nv_bfloat16>() {
    return GGML_TYPE_BF16;
}

struct tensor_dims {
    int64_t head_dim;
    int64_t seq_len;
    int64_t heads;
    int64_t batch;
};

struct tensor_strides {
    int64_t stride_seq;
    int64_t stride_head;
    int64_t stride_batch;
};

struct quant_buffers {
    int8_t * data = nullptr;
    float  * scale = nullptr;
};

struct pv_buffers {
    int8_t * data = nullptr;
    float  * scale = nullptr;
};

constexpr int CTA_Q = 128;
constexpr int CTA_K = 64;
constexpr int WARP_Q = 32;
constexpr int WARP_K = 64;
constexpr int BLKQ   = 128;
constexpr int WARPQ  = 32;
constexpr int BLKK   = 64;
constexpr int WARPK  = 64;
constexpr int CTA_TRANSPOSE = 64;

inline tensor_dims get_tensor_dims(const ggml_tensor * t) {
    return {
        /*head_dim*/ t->ne[0],
        /*seq_len */ t->ne[1],
        /*heads   */ t->ne[2],
        /*batch   */ t->ne[3],
    };
}

inline tensor_strides get_tensor_strides(const ggml_tensor * t) {
    const size_t ts = ggml_type_size(t->type);
    return {
        /*stride_seq  */ t->nb[1] / ts,
        /*stride_head */ t->nb[2] / ts,
        /*stride_batch*/ t->nb[3] / ts,
    };
}

inline tensor_strides make_contiguous_strides(const tensor_dims & dims) {
    return {
        /*stride_seq  */ dims.head_dim,
        /*stride_head */ dims.head_dim * dims.seq_len,
        /*stride_batch*/ dims.head_dim * dims.seq_len * dims.heads,
    };
}

inline tensor_strides make_scale_strides(int heads, int cols) {
    return {
        /*stride_seq  */ 0,
        /*stride_head */ cols,
        /*stride_batch*/ heads * cols,
    };
}

struct value_strides {
    uint32_t stride_bz;
    uint32_t stride_h;
    uint32_t stride_d;
};

inline value_strides make_value_strides(int heads, int head_dim, int seq_len) {
    const int64_t stride_d = seq_len;
    const int64_t stride_h = head_dim * stride_d;
    const int64_t stride_bz = heads * stride_h;
    GGML_ASSERT(stride_d <= std::numeric_limits<uint32_t>::max());
    GGML_ASSERT(stride_h <= std::numeric_limits<uint32_t>::max());
    GGML_ASSERT(stride_bz <= std::numeric_limits<uint32_t>::max());
    return {
        (uint32_t) stride_bz,
        (uint32_t) stride_h,
        (uint32_t) stride_d,
    };
}

inline uint32_t to_u32(int64_t v) {
    GGML_ASSERT(v >= 0 && v <= std::numeric_limits<uint32_t>::max());
    return (uint32_t) v;
}

constexpr float FP8_SCALE_MAX = 448.0f;

template<typename T>
__global__ void copy_pad_tensor_kernel(
        const T * __restrict__ src,
        T * __restrict__ dst,
        int head_dim_src,
        int head_dim_dst,
        int64_t seq_len_src,
        int64_t seq_len_dst,
        int64_t heads,
        int64_t batch,
        int64_t stride_seq_src,
        int64_t stride_head_src,
        int64_t stride_batch_src,
        int64_t stride_seq_dst,
        int64_t stride_head_dst,
        int64_t stride_batch_dst) {
    const int64_t row = blockIdx.x;
    const int lane = threadIdx.x;
    const int64_t total_rows = seq_len_dst * heads * batch;
    if (row >= total_rows) {
        return;
    }

    int64_t tmp = row;
    const int64_t seq = tmp % seq_len_dst;
    tmp /= seq_len_dst;
    const int64_t head = tmp % heads;
    const int64_t batch_idx = tmp / heads;

    const bool within_seq = seq < seq_len_src;

    const T * src_row = src +
        batch_idx*stride_batch_src +
        head*stride_head_src +
        (within_seq ? seq*stride_seq_src : 0);

    T * dst_row = dst +
        batch_idx*stride_batch_dst +
        head*stride_head_dst +
        seq*stride_seq_dst;

    for (int d = lane; d < head_dim_dst; d += blockDim.x) {
        if (within_seq && d < head_dim_src) {
            dst_row[d] = src_row[d];
        } else {
            dst_row[d] = T(0);
        }
    }
}

template<typename T>
__global__ void trim_tensor_kernel(
        const T * __restrict__ src,
        T * __restrict__ dst,
        int head_dim_dst,
        int head_dim_src,
        int64_t seq_len,
        int64_t heads,
        int64_t batch,
        int64_t stride_seq_src,
        int64_t stride_head_src,
        int64_t stride_batch_src,
        int64_t stride_seq_dst,
        int64_t stride_head_dst,
        int64_t stride_batch_dst) {
    const int64_t row = blockIdx.x;
    const int lane = threadIdx.x;
    const int64_t total_rows = seq_len * heads * batch;
    if (row >= total_rows) {
        return;
    }

    int64_t tmp = row;
    const int64_t seq = tmp % seq_len;
    tmp /= seq_len;
    const int64_t head = tmp % heads;
    const int64_t batch_idx = tmp / heads;

    const T * src_row = src +
        batch_idx*stride_batch_src +
        head*stride_head_src +
        seq*stride_seq_src;

    T * dst_row = dst +
        batch_idx*stride_batch_dst +
        head*stride_head_dst +
        seq*stride_seq_dst;

    for (int d = lane; d < head_dim_dst; d += blockDim.x) {
        dst_row[d] = src_row[d];
    }
}

template<typename T>
__global__ void compute_k_mean_kernel(
        const T * __restrict__ src,
        T * __restrict__ dst,
        int head_dim,
        int64_t seq_len,
        int64_t heads,
        int64_t batch,
        int64_t stride_seq,
        int64_t stride_head,
        int64_t stride_batch,
        int64_t stride_head_dst,
        int64_t stride_batch_dst) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = (int64_t) head_dim * heads * batch;
    if (idx >= total) {
        return;
    }

    int64_t tmp = idx;
    const int dim = tmp % head_dim;
    tmp /= head_dim;
    const int head = tmp % heads;
    const int batch_idx = tmp / heads;

    const T * base = src +
        batch_idx*stride_batch +
        head*stride_head +
        dim;

    float sum = 0.0f;
    for (int64_t s = 0; s < seq_len; ++s) {
        sum += ggml_cuda_sage::convert_to_float(base[s*stride_seq]);
    }
    const float mean = sum / (float) seq_len;
    dst[batch_idx*stride_batch_dst + head*stride_head_dst + dim] =
        ggml_cuda_sage::convert_from_float<T>(mean);
}

inline int64_t div_ceil64(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

template<typename T>
void copy_pad_tensor(
        const T * src,
        const tensor_dims & dims_src,
        const tensor_strides & strides_src,
        const tensor_dims & dims_dst,
        const tensor_strides & strides_dst,
        T * dst,
        cudaStream_t stream) {
    const int64_t total_rows = dims_dst.seq_len * dims_dst.heads * dims_dst.batch;
    const int threads = 128;
    copy_pad_tensor_kernel<T><<<total_rows, threads, 0, stream>>>(
        src, dst,
        dims_src.head_dim, dims_dst.head_dim,
        dims_src.seq_len, dims_dst.seq_len,
        dims_dst.heads, dims_dst.batch,
        strides_src.stride_seq, strides_src.stride_head, strides_src.stride_batch,
        strides_dst.stride_seq, strides_dst.stride_head, strides_dst.stride_batch);
    CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void trim_tensor_head_dim(
        const T * src,
        int head_dim_src,
        const tensor_dims & dims,
        const tensor_strides & strides_src,
        const tensor_strides & strides_dst,
        T * dst,
        cudaStream_t stream) {
    const int64_t total_rows = dims.seq_len * dims.heads * dims.batch;
    const int threads = 128;
    trim_tensor_kernel<T><<<total_rows, threads, 0, stream>>>(
        src, dst,
        dims.head_dim, head_dim_src,
        dims.seq_len, dims.heads, dims.batch,
        strides_src.stride_seq, strides_src.stride_head, strides_src.stride_batch,
        strides_dst.stride_seq, strides_dst.stride_head, strides_dst.stride_batch);
    CUDA_CHECK(cudaGetLastError());
}

template<typename T, int HEAD_DIM>
void quantize_q_per_warp(
        const T * input,
        int8_t * output,
        float * scale,
        int batch,
        int heads,
        int seq_len,
        cudaStream_t stream) {
    constexpr int BLOCK_SIZE = BLKQ;
    constexpr int WARP_BLOCK_SIZE = WARPQ;
    constexpr int num_pack_per_thread = (WARP_BLOCK_SIZE * (HEAD_DIM / 8) + 1023) / 1024;

    const int scale_cols = ((seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE) * (BLOCK_SIZE / WARP_BLOCK_SIZE);

    const int64_t stride_seq = HEAD_DIM;
    const int64_t stride_head = HEAD_DIM * seq_len;
    const int64_t stride_batch = stride_head * heads;

    const int64_t stride_head_scale = scale_cols;
    const int64_t stride_batch_scale = heads * scale_cols;

    dim3 grid(((seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE) * (BLOCK_SIZE / WARP_BLOCK_SIZE), heads, batch);
    dim3 block(WARP_BLOCK_SIZE * (HEAD_DIM / 8) / num_pack_per_thread);

    ggml_cuda_sage::quant_int8_kernel<HEAD_DIM, WARP_BLOCK_SIZE, num_pack_per_thread, false, false, T>
        <<<grid, block, 0, stream>>>(
            const_cast<T *>(input),
            nullptr,
            output,
            scale,
            0.0f,
            seq_len,
            to_u32(stride_batch), to_u32(stride_seq), to_u32(stride_head),
            0, 0,
            to_u32(stride_batch), to_u32(stride_seq), to_u32(stride_head),
            to_u32(stride_batch_scale), to_u32(stride_head_scale));
    CUDA_CHECK(cudaGetLastError());
}

template<typename T, int HEAD_DIM, bool SUB_MEAN>
void quantize_k_per_block(
        const T * input,
        const T * mean,
        int8_t * output,
        float * scale,
        int batch,
        int heads,
        int seq_len,
        cudaStream_t stream) {
    constexpr int BLOCK_SIZE = BLKK;
    constexpr int num_pack_per_thread = (BLOCK_SIZE * (HEAD_DIM / 8) + 1023) / 1024;

    const int scale_cols = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    const int64_t stride_seq = HEAD_DIM;
    const int64_t stride_head = HEAD_DIM * seq_len;
    const int64_t stride_batch = stride_head * heads;

    const int64_t stride_head_scale = scale_cols;
    const int64_t stride_batch_scale = heads * scale_cols;

    const int64_t stride_head_mean = SUB_MEAN ? HEAD_DIM : 0;
    const int64_t stride_batch_mean = SUB_MEAN ? HEAD_DIM * heads : 0;

    dim3 grid((seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE, heads, batch);
    dim3 block(BLOCK_SIZE * (HEAD_DIM / 8) / num_pack_per_thread);

    ggml_cuda_sage::quant_int8_kernel<HEAD_DIM, BLOCK_SIZE, num_pack_per_thread, false, SUB_MEAN, T>
        <<<grid, block, 0, stream>>>(
            const_cast<T *>(input),
            const_cast<T *>(mean),
            output,
            scale,
            0.0f,
            seq_len,
            to_u32(stride_batch), to_u32(stride_seq), to_u32(stride_head),
            to_u32(stride_batch_mean), to_u32(stride_head_mean),
            to_u32(stride_batch), to_u32(stride_seq), to_u32(stride_head),
            to_u32(stride_batch_scale), to_u32(stride_head_scale));
    CUDA_CHECK(cudaGetLastError());
}

__device__ __forceinline__ void atomic_max_relaxed(float * addr, float val) {
    if (val <= 0.0f) {
        return;
    }
    int * addr_i = reinterpret_cast<int *>(addr);
    int old = __float_as_int(*addr);
    while (__int_as_float(old) < val) {
        const int assumed = old;
        old = atomicCAS(addr_i, assumed, __float_as_int(val));
        if (assumed == old) {
            break;
        }
    }
}

template<typename T, int HEAD_DIM>
__global__ void quantize_q_per_thread_kernel(
        const T * __restrict__ input,
        int8_t * __restrict__ output,
        float * __restrict__ scale,
        int seq_len,
        uint32_t stride_batch,
        uint32_t stride_seq,
        uint32_t stride_head,
        uint32_t stride_batch_scale,
        uint32_t stride_head_scale) {
    constexpr int TOKENS_PER_BLOCK = WARPQ;
    constexpr int NUM_GROUPS = 8;
    constexpr int PACK_SIZE = 8;
    constexpr int THREADS_PER_TOKEN = HEAD_DIM / PACK_SIZE;

    const int thread_id = threadIdx.x;
    if (thread_id >= THREADS_PER_TOKEN * TOKENS_PER_BLOCK) {
        return;
    }

    const int token_local = thread_id / THREADS_PER_TOKEN;
    const int lane_offset = (thread_id % THREADS_PER_TOKEN) * PACK_SIZE;
    const int group = token_local % NUM_GROUPS;

    const int block_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int batch_idx = blockIdx.z;

    const int token_index = block_idx * TOKENS_PER_BLOCK + token_local;
    const bool token_valid = token_index < seq_len;

    float vals[PACK_SIZE];
    float local_max = 0.0f;

    if (token_valid) {
        const size_t base_offset = (size_t) batch_idx * stride_batch +
                                   (size_t) head_idx  * stride_head  +
                                   (size_t) token_index * stride_seq +
                                   lane_offset;
        const T * src = input + base_offset;
#pragma unroll
        for (int i = 0; i < PACK_SIZE; ++i) {
            vals[i] = ggml_cuda_sage::convert_to_float(src[i]);
            local_max = fmaxf(local_max, fabsf(vals[i]));
        }
    } else {
#pragma unroll
        for (int i = 0; i < PACK_SIZE; ++i) {
            vals[i] = 0.0f;
        }
    }

    __shared__ float s_group_max[NUM_GROUPS];
    if (thread_id < NUM_GROUPS) {
        s_group_max[thread_id] = 1e-7f;
    }
    __syncthreads();

    atomic_max_relaxed(&s_group_max[group], fmaxf(local_max, 1e-7f));
    __syncthreads();

    float group_max = fmaxf(s_group_max[group], 1e-7f);
    const float inv_scale = 127.0f / group_max;
    const float scale_val = group_max / 127.0f;

    char4 packed[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const float v0 = vals[j * 4 + 0] * inv_scale;
        const float v1 = vals[j * 4 + 1] * inv_scale;
        const float v2 = vals[j * 4 + 2] * inv_scale;
        const float v3 = vals[j * 4 + 3] * inv_scale;
        packed[j] = make_char4(
            float_to_int8_rn(v0),
            float_to_int8_rn(v1),
            float_to_int8_rn(v2),
            float_to_int8_rn(v3));
    }

    if (token_valid) {
        const size_t base_offset = (size_t) batch_idx * stride_batch +
                                   (size_t) head_idx  * stride_head  +
                                   (size_t) token_index * stride_seq +
                                   lane_offset;
        int8_t * dst = output + base_offset;
        *reinterpret_cast<float2 *>(dst) = *reinterpret_cast<const float2 *>(&packed[0]);
    }

    const size_t scale_base = (size_t) batch_idx * stride_batch_scale +
                              (size_t) head_idx  * stride_head_scale +
                              (size_t) block_idx * NUM_GROUPS;
    if (token_local == group && (thread_id % THREADS_PER_TOKEN) == 0) {
        scale[scale_base + group] = scale_val;
    }
}

template<typename T, int HEAD_DIM, bool SUB_MEAN>
__global__ void quantize_k_per_thread_kernel(
        const T * __restrict__ input,
        const T * __restrict__ mean,
        int8_t * __restrict__ output,
        float * __restrict__ scale,
        int seq_len,
        uint32_t stride_batch,
        uint32_t stride_seq,
        uint32_t stride_head,
        uint32_t stride_batch_mean,
        uint32_t stride_head_mean,
        uint32_t stride_batch_scale,
        uint32_t stride_head_scale) {
    constexpr int TOKENS_PER_BLOCK = WARPK;
    constexpr int NUM_GROUPS = 4;
    constexpr int PACK_SIZE = 8;
    constexpr int THREADS_PER_TOKEN = HEAD_DIM / PACK_SIZE;

    const int thread_id = threadIdx.x;
    if (thread_id >= THREADS_PER_TOKEN * TOKENS_PER_BLOCK) {
        return;
    }

    const int token_local = thread_id / THREADS_PER_TOKEN;
    const int lane_offset = (thread_id % THREADS_PER_TOKEN) * PACK_SIZE;
    const int group = (token_local % 8) >> 1;

    const int block_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int batch_idx = blockIdx.z;

    const int token_index = block_idx * TOKENS_PER_BLOCK + token_local;
    const bool token_valid = token_index < seq_len;

    float vals[PACK_SIZE];
    float mean_vals[PACK_SIZE];

    if constexpr (SUB_MEAN) {
        const size_t mean_offset = (size_t) batch_idx * stride_batch_mean +
                                   (size_t) head_idx  * stride_head_mean +
                                   lane_offset;
        const T * mean_ptr = mean + mean_offset;
#pragma unroll
        for (int i = 0; i < PACK_SIZE; ++i) {
            mean_vals[i] = ggml_cuda_sage::convert_to_float(mean_ptr[i]);
        }
    }

    float local_max = 0.0f;
    if (token_valid) {
        const size_t base_offset = (size_t) batch_idx * stride_batch +
                                   (size_t) head_idx  * stride_head  +
                                   (size_t) token_index * stride_seq +
                                   lane_offset;
        const T * src = input + base_offset;
#pragma unroll
        for (int i = 0; i < PACK_SIZE; ++i) {
            vals[i] = ggml_cuda_sage::convert_to_float(src[i]);
            if constexpr (SUB_MEAN) {
                vals[i] -= mean_vals[i];
            }
            local_max = fmaxf(local_max, fabsf(vals[i]));
        }
    } else {
#pragma unroll
        for (int i = 0; i < PACK_SIZE; ++i) {
            vals[i] = 0.0f;
        }
    }

    __shared__ float s_group_max[NUM_GROUPS];
    if (thread_id < NUM_GROUPS) {
        s_group_max[thread_id] = 1e-7f;
    }
    __syncthreads();

    atomic_max_relaxed(&s_group_max[group], fmaxf(local_max, 1e-7f));
    __syncthreads();

    float group_max = fmaxf(s_group_max[group], 1e-7f);
    const float inv_scale = 127.0f / group_max;
    const float scale_val = group_max / 127.0f;

    char4 packed[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const float v0 = vals[j * 4 + 0] * inv_scale;
        const float v1 = vals[j * 4 + 1] * inv_scale;
        const float v2 = vals[j * 4 + 2] * inv_scale;
        const float v3 = vals[j * 4 + 3] * inv_scale;
        packed[j] = make_char4(
            float_to_int8_rn(v0),
            float_to_int8_rn(v1),
            float_to_int8_rn(v2),
            float_to_int8_rn(v3));
    }

    if (token_valid) {
        const size_t base_offset = (size_t) batch_idx * stride_batch +
                                   (size_t) head_idx  * stride_head  +
                                   (size_t) token_index * stride_seq +
                                   lane_offset;
        int8_t * dst = output + base_offset;
        *reinterpret_cast<float2 *>(dst) = *reinterpret_cast<const float2 *>(&packed[0]);
    }

    const size_t scale_base = (size_t) batch_idx * stride_batch_scale +
                              (size_t) head_idx  * stride_head_scale +
                              (size_t) block_idx * NUM_GROUPS;
    if (token_local == group * 2 && (thread_id % THREADS_PER_TOKEN) == 0) {
        scale[scale_base + group] = scale_val;
    }
}

template<typename T, int HEAD_DIM>
void quantize_q_per_thread(
        const T * input,
        int8_t * output,
        float * scale,
        int batch,
        int heads,
        int seq_len,
        cudaStream_t stream) {
    constexpr int CTA_SIZE = CTA_Q;
    constexpr int TOKENS_PER_BLOCK = WARPQ;
    const int num_blocks = div_ceil64(seq_len, CTA_SIZE) * (CTA_SIZE / TOKENS_PER_BLOCK);

    const int64_t stride_seq = HEAD_DIM;
    const int64_t stride_head = HEAD_DIM * seq_len;
    const int64_t stride_batch = stride_head * heads;

    const int64_t stride_head_scale = (int64_t) num_blocks * 8;
    const int64_t stride_batch_scale = heads * stride_head_scale;

    dim3 grid(num_blocks, heads, batch);
    dim3 block(TOKENS_PER_BLOCK * (HEAD_DIM / 8));

    quantize_q_per_thread_kernel<T, HEAD_DIM><<<grid, block, 0, stream>>>(
        input,
        output,
        scale,
        seq_len,
        to_u32(stride_batch), to_u32(stride_seq), to_u32(stride_head),
        to_u32(stride_batch_scale), to_u32(stride_head_scale));
    CUDA_CHECK(cudaGetLastError());
}

template<typename T, int HEAD_DIM, bool SUB_MEAN>
void quantize_k_per_thread(
        const T * input,
        const T * mean,
        int8_t * output,
        float * scale,
        int batch,
        int heads,
        int seq_len,
        cudaStream_t stream) {
    constexpr int CTA_SIZE = CTA_K;
    constexpr int TOKENS_PER_BLOCK = WARPK;
    const int num_blocks = div_ceil64(seq_len, CTA_SIZE);

    const int64_t stride_seq = HEAD_DIM;
    const int64_t stride_head = HEAD_DIM * seq_len;
    const int64_t stride_batch = stride_head * heads;

    const int64_t stride_head_scale = (int64_t) num_blocks * 4;
    const int64_t stride_batch_scale = heads * stride_head_scale;

    const int64_t stride_head_mean = SUB_MEAN ? HEAD_DIM : 0;
    const int64_t stride_batch_mean = SUB_MEAN ? HEAD_DIM * heads : 0;

    dim3 grid(num_blocks, heads, batch);
    dim3 block(TOKENS_PER_BLOCK * (HEAD_DIM / 8));

    quantize_k_per_thread_kernel<T, HEAD_DIM, SUB_MEAN><<<grid, block, 0, stream>>>(
        input,
        mean,
        output,
        scale,
        seq_len,
        to_u32(stride_batch), to_u32(stride_seq), to_u32(stride_head),
        to_u32(stride_batch_mean), to_u32(stride_head_mean),
        to_u32(stride_batch_scale), to_u32(stride_head_scale));
    CUDA_CHECK(cudaGetLastError());
}

template<typename T, int HEAD_DIM>
void transpose_pad_permute(
        const T * input,
        T * output,
        int batch,
        int heads,
        int seq_len,
        int padded_seq_len,
        cudaStream_t stream) {
    constexpr int CTA_SIZE = CTA_TRANSPOSE;
    const int padded = ((seq_len + CTA_SIZE - 1) / CTA_SIZE) * CTA_SIZE;
    GGML_ASSERT(padded == padded_seq_len);

    const int64_t stride_seq_in = HEAD_DIM;
    const int64_t stride_head_in = HEAD_DIM * seq_len;
    const int64_t stride_batch_in = stride_head_in * heads;

    const int64_t stride_d_out = padded_seq_len;
    const int64_t stride_head_out = HEAD_DIM * stride_d_out;
    const int64_t stride_batch_out = heads * stride_head_out;

    dim3 grid(padded_seq_len / CTA_SIZE, heads, batch);
    dim3 block(CTA_SIZE * (HEAD_DIM / 8));

    ggml_cuda_sage::transpose_pad_permute_kernel<HEAD_DIM, CTA_SIZE, true, T>
        <<<grid, block, 0, stream>>>(
            const_cast<T *>(input),
            output,
            seq_len,
            to_u32(stride_batch_in), to_u32(stride_seq_in), to_u32(stride_head_in),
            to_u32(stride_batch_out), to_u32(stride_d_out), to_u32(stride_head_out));
    CUDA_CHECK(cudaGetLastError());
}

template<typename T, int HEAD_DIM, bool SUB_MEAN>
void quantize_v_to_fp8(
        T * input,
        int8_t * output,
        float * mean,
        float * scale,
        float scale_max,
        int batch,
        int heads,
        int head_dim,
        int seq_len,
        cudaStream_t stream) {
    constexpr int CTA_SIZE = 256;

    const int64_t stride_d = seq_len;
    const int64_t stride_head = head_dim * stride_d;
    const int64_t stride_batch = heads * stride_head;

    const int64_t stride_head_scale = head_dim;
    const int64_t stride_batch_scale = heads * head_dim;

    dim3 grid(heads, batch, head_dim);
    dim3 block(CTA_SIZE);

    ggml_cuda_sage::mean_scale_kernel<64, SUB_MEAN, T><<<grid, block, 0, stream>>>(
        input,
        output,
        mean,
        scale,
        scale_max,
        seq_len,
        to_u32(stride_batch), to_u32(stride_d), to_u32(stride_head),
        to_u32(stride_batch), to_u32(stride_d), to_u32(stride_head),
        SUB_MEAN ? to_u32(stride_batch_scale) : 0,
        SUB_MEAN ? to_u32(stride_head_scale) : 0,
        to_u32(stride_batch_scale), to_u32(stride_head_scale));
    CUDA_CHECK(cudaGetLastError());
}

template<typename T, int HEAD_DIM, MaskMode MODE, QuantGranularity Q_GRAN, QuantGranularity K_GRAN, bool FUSE_V_MEAN>
void launch_sage_kernel_variant(
        const int8_t * q,
        const int8_t * k,
        const int8_t * v,
        T * output,
        const float * q_scale,
        const float * k_scale,
        const float * v_scale,
        const float * v_mean,
        int batch,
        int num_q_heads,
        int num_k_heads,
        int qo_len,
        int kv_len,
        int num_kv_groups,
        const tensor_strides & q_strides,
        const tensor_strides & k_strides,
        const tensor_strides & o_strides,
        const value_strides & v_strides,
        float sm_scale,
        cudaStream_t stream) {
    constexpr int CTA_Q = 128;
    constexpr int CTA_K = 64;
    constexpr int WARP_Q = 32;
    constexpr int WARP_K = 64;

    dim3 grid(div_ceil64(qo_len, CTA_Q), num_q_heads, batch);
    dim3 block(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));

    const size_t smem_q = (size_t) CTA_Q * HEAD_DIM * sizeof(int8_t);
    const size_t smem_k = (size_t) CTA_K * HEAD_DIM * sizeof(int8_t);
    const size_t smem_v = (size_t) CTA_K * HEAD_DIM * sizeof(int8_t);
    const size_t smem_o = (size_t) CTA_Q * HEAD_DIM * sizeof(half);
    const size_t smem_max = std::max(smem_q + smem_k + smem_v, smem_o);

    auto kernel_func =
        qk_int_sv_f8_attn_kernel<
            CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM,
            DataType::kInt8,
            Q_GRAN,
            K_GRAN,
            float,
            true,
            T,
            ComputeUnit::kCudaCore,
            MODE,
            false,
            true,
            FUSE_V_MEAN,
            true>;

    CUDA_CHECK(cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max));

    kernel_func<<<grid, block, smem_max, stream>>>(
        const_cast<int8_t *>(q),
        const_cast<int8_t *>(k),
        const_cast<int8_t *>(v),
        output,
        nullptr,
        const_cast<float *>(q_scale),
        const_cast<float *>(k_scale),
        const_cast<float *>(v_scale),
        FUSE_V_MEAN ? const_cast<float *>(v_mean) : nullptr,
        qo_len,
        kv_len,
        num_kv_groups,
        to_u32(q_strides.stride_batch), to_u32(q_strides.stride_seq), to_u32(q_strides.stride_head),
        to_u32(k_strides.stride_batch), to_u32(k_strides.stride_seq), to_u32(k_strides.stride_head),
        v_strides.stride_bz, v_strides.stride_h, v_strides.stride_d,
        to_u32(o_strides.stride_batch), to_u32(o_strides.stride_seq), to_u32(o_strides.stride_head),
        sm_scale);
    CUDA_CHECK(cudaGetLastError());
}

template<typename T, int HEAD_DIM, MaskMode MODE>
void launch_sage_kernel(
        const int8_t * q,
        const int8_t * k,
        const int8_t * v,
        T * output,
        const float * q_scale,
        const float * k_scale,
        const float * v_scale,
        const float * v_mean,
        int batch,
        int num_q_heads,
        int num_k_heads,
        int qo_len,
        int kv_len,
        int num_kv_groups,
        const tensor_strides & q_strides,
        const tensor_strides & k_strides,
        const tensor_strides & o_strides,
        const value_strides & v_strides,
        float sm_scale,
        cudaStream_t stream,
        ggml_sage_qk_granularity granularity) {
    const bool fuse_v_mean = v_mean != nullptr;

    if (granularity == GGML_SAGE_QK_GRANULARITY_PER_THREAD) {
        if (fuse_v_mean) {
            launch_sage_kernel_variant<T, HEAD_DIM, MODE, QuantGranularity::kPerThread, QuantGranularity::kPerThread, true>(
                q, k, v, output, q_scale, k_scale, v_scale, v_mean,
                batch, num_q_heads, num_k_heads, qo_len, kv_len, num_kv_groups,
                q_strides, k_strides, o_strides, v_strides, sm_scale, stream);
        } else {
            launch_sage_kernel_variant<T, HEAD_DIM, MODE, QuantGranularity::kPerThread, QuantGranularity::kPerThread, false>(
                q, k, v, output, q_scale, k_scale, v_scale, nullptr,
                batch, num_q_heads, num_k_heads, qo_len, kv_len, num_kv_groups,
                q_strides, k_strides, o_strides, v_strides, sm_scale, stream);
        }
    } else {
        if (fuse_v_mean) {
            launch_sage_kernel_variant<T, HEAD_DIM, MODE, QuantGranularity::kPerWarp, QuantGranularity::kPerWarp, true>(
                q, k, v, output, q_scale, k_scale, v_scale, v_mean,
                batch, num_q_heads, num_k_heads, qo_len, kv_len, num_kv_groups,
                q_strides, k_strides, o_strides, v_strides, sm_scale, stream);
        } else {
            launch_sage_kernel_variant<T, HEAD_DIM, MODE, QuantGranularity::kPerWarp, QuantGranularity::kPerWarp, false>(
                q, k, v, output, q_scale, k_scale, v_scale, nullptr,
                batch, num_q_heads, num_k_heads, qo_len, kv_len, num_kv_groups,
                q_strides, k_strides, o_strides, v_strides, sm_scale, stream);
        }
    }
}

template<typename T>
void compute_k_mean(
        const T * src,
        const tensor_dims & dims,
        const tensor_strides & strides,
        const tensor_strides & mean_strides,
        T * mean,
        cudaStream_t stream) {
    const int64_t total = (int64_t) dims.head_dim * dims.heads * dims.batch;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    compute_k_mean_kernel<T><<<blocks, threads, 0, stream>>>(
        src, mean,
        dims.head_dim,
        dims.seq_len, dims.heads, dims.batch,
        strides.stride_seq,
        strides.stride_head,
        strides.stride_batch,
        mean_strides.stride_head,
        mean_strides.stride_batch);
    CUDA_CHECK(cudaGetLastError());
}

template<typename T>
struct type_traits;

template<>
struct type_traits<half> {
    static constexpr ggml_type ggml = GGML_TYPE_F16;
};

template<>
struct type_traits<nv_bfloat16> {
    static constexpr ggml_type ggml = GGML_TYPE_BF16;
};

template<typename T>
void sage_attn_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(dst->src[0] && dst->src[1] && dst->src[2]);

    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];

    const float softmax_scale = ggml_get_op_params_f32(dst, 0);
    const bool  is_causal     = ggml_get_op_params_i32(dst, 1) != 0;
    const bool  smooth_k      = ggml_get_op_params_i32(dst, 2) != 0;
    const auto  granularity   = static_cast<ggml_sage_qk_granularity>(ggml_get_op_params_i32(dst, 3));
    const bool  per_thread_qk = (granularity == GGML_SAGE_QK_GRANULARITY_PER_THREAD);

    GGML_UNUSED(is_causal);

    const tensor_dims q_dims = get_tensor_dims(q);
    const tensor_dims k_dims = get_tensor_dims(k);
    const tensor_dims v_dims = get_tensor_dims(v);

    GGML_ASSERT(q_dims.head_dim == k_dims.head_dim);

    const int head_dim = q_dims.head_dim;
    int head_dim_padded = 0;
    if (head_dim <= 64) {
        head_dim_padded = 64;
    } else if (head_dim <= 128) {
        head_dim_padded = 128;
    } else {
        GGML_ABORT("SageAttention SM89: unsupported head_dim (must be <= 128)");
    }

    const int64_t qo_len = q_dims.seq_len;
    const int64_t kv_len = k_dims.seq_len;
    const int64_t kv_len_padded = div_ceil64(kv_len, CTA_K) * CTA_K;

    const tensor_dims q_pad_dims = { head_dim_padded, qo_len, q_dims.heads, q_dims.batch };
    const tensor_dims k_pad_dims = { head_dim_padded, kv_len, k_dims.heads, k_dims.batch };
    const tensor_dims v_pad_dims = { head_dim_padded, kv_len_padded, v_dims.heads, v_dims.batch };

    const tensor_strides q_strides = get_tensor_strides(q);
    const tensor_strides k_strides = get_tensor_strides(k);
    const tensor_strides v_strides = get_tensor_strides(v);

    const tensor_strides q_pad_strides = make_contiguous_strides(q_pad_dims);
    const tensor_strides k_pad_strides = make_contiguous_strides(k_pad_dims);
    const tensor_strides v_pad_strides = make_contiguous_strides(v_pad_dims);

    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<T> q_padded(pool, q_pad_dims.head_dim * q_pad_dims.seq_len * q_pad_dims.heads * q_pad_dims.batch);
    ggml_cuda_pool_alloc<T> k_padded(pool, k_pad_dims.head_dim * k_pad_dims.seq_len * k_pad_dims.heads * k_pad_dims.batch);
    ggml_cuda_pool_alloc<T> v_padded(pool, v_pad_dims.head_dim * v_pad_dims.seq_len * v_pad_dims.heads * v_pad_dims.batch);

    copy_pad_tensor(
        static_cast<const T *>(q->data),
        q_dims, q_strides,
        q_pad_dims, q_pad_strides,
        q_padded.get(),
        stream);

    copy_pad_tensor(
        static_cast<const T *>(k->data),
        k_dims, k_strides,
        k_pad_dims, k_pad_strides,
        k_padded.get(),
        stream);

    copy_pad_tensor(
        static_cast<const T *>(v->data),
        v_dims, v_strides,
        v_pad_dims, v_pad_strides,
        v_padded.get(),
        stream);

    const int batch = q_dims.batch;
    const int num_q_heads = q_dims.heads;
    const int num_k_heads = k_dims.heads;
    GGML_ASSERT(num_q_heads % num_k_heads == 0);
    const int num_kv_groups = num_q_heads / num_k_heads;

    const int q_blocks = div_ceil64(qo_len, CTA_Q) * (CTA_Q / WARPQ);
    const int k_blocks = div_ceil64(kv_len, CTA_K) * (CTA_K / WARP_K);
    const int q_scale_cols = per_thread_qk ? q_blocks * 8 : ((qo_len + BLKQ - 1) / BLKQ) * (BLKQ / WARPQ);
    const int k_scale_cols = per_thread_qk ? k_blocks * 4 : (kv_len + BLKK - 1) / BLKK;

    ggml_cuda_pool_alloc<int8_t> q_int8(pool, q_pad_dims.head_dim * qo_len * num_q_heads * batch);
    ggml_cuda_pool_alloc<int8_t> k_int8(pool, k_pad_dims.head_dim * kv_len * num_k_heads * batch);
    ggml_cuda_pool_alloc<float>  q_scale(pool, (size_t) batch * num_q_heads * q_scale_cols);
    ggml_cuda_pool_alloc<float>  k_scale(pool, (size_t) batch * num_k_heads * k_scale_cols);

    switch (head_dim_padded) {
        case 64:
            if (per_thread_qk) {
                quantize_q_per_thread<T, 64>(q_padded.get(), q_int8.get(), q_scale.get(), batch, num_q_heads, qo_len, stream);
            } else {
                quantize_q_per_warp<T, 64>(q_padded.get(), q_int8.get(), q_scale.get(), batch, num_q_heads, qo_len, stream);
            }
            break;
        case 128:
            if (per_thread_qk) {
                quantize_q_per_thread<T, 128>(q_padded.get(), q_int8.get(), q_scale.get(), batch, num_q_heads, qo_len, stream);
            } else {
                quantize_q_per_warp<T, 128>(q_padded.get(), q_int8.get(), q_scale.get(), batch, num_q_heads, qo_len, stream);
            }
            break;
        default:
            GGML_ABORT("SageAttention SM89: unsupported head_dim");
    }

    if (smooth_k) {
        ggml_cuda_pool_alloc<T> k_mean(pool, (size_t) head_dim_padded * num_k_heads * batch);
        tensor_strides mean_strides = {
            0,
            head_dim_padded,
            head_dim_padded * num_k_heads,
        };
        compute_k_mean(k_padded.get(), k_pad_dims, k_pad_strides, mean_strides, k_mean.get(), stream);
        switch (head_dim_padded) {
            case 64:
                if (per_thread_qk) {
                    quantize_k_per_thread<T, 64, true>(k_padded.get(), k_mean.get(), k_int8.get(), k_scale.get(), batch, num_k_heads, kv_len, stream);
                } else {
                    quantize_k_per_block<T, 64, true>(k_padded.get(), k_mean.get(), k_int8.get(), k_scale.get(), batch, num_k_heads, kv_len, stream);
                }
                break;
            case 128:
                if (per_thread_qk) {
                    quantize_k_per_thread<T, 128, true>(k_padded.get(), k_mean.get(), k_int8.get(), k_scale.get(), batch, num_k_heads, kv_len, stream);
                } else {
                    quantize_k_per_block<T, 128, true>(k_padded.get(), k_mean.get(), k_int8.get(), k_scale.get(), batch, num_k_heads, kv_len, stream);
                }
                break;
        }
    } else {
        switch (head_dim_padded) {
            case 64:
                if (per_thread_qk) {
                    quantize_k_per_thread<T, 64, false>(k_padded.get(), nullptr, k_int8.get(), k_scale.get(), batch, num_k_heads, kv_len, stream);
                } else {
                    quantize_k_per_block<T, 64, false>(k_padded.get(), nullptr, k_int8.get(), k_scale.get(), batch, num_k_heads, kv_len, stream);
                }
                break;
            case 128:
                if (per_thread_qk) {
                    quantize_k_per_thread<T, 128, false>(k_padded.get(), nullptr, k_int8.get(), k_scale.get(), batch, num_k_heads, kv_len, stream);
                } else {
                    quantize_k_per_block<T, 128, false>(k_padded.get(), nullptr, k_int8.get(), k_scale.get(), batch, num_k_heads, kv_len, stream);
                }
                break;
        }
    }

    ggml_cuda_pool_alloc<T> v_transposed(pool, (size_t) batch * num_k_heads * head_dim_padded * kv_len_padded);
    switch (head_dim_padded) {
        case 64:
            transpose_pad_permute<T, 64>(v_padded.get(), v_transposed.get(), batch, num_k_heads, kv_len, kv_len_padded, stream);
            break;
        case 128:
            transpose_pad_permute<T, 128>(v_padded.get(), v_transposed.get(), batch, num_k_heads, kv_len, kv_len_padded, stream);
            break;
    }

    ggml_cuda_pool_alloc<int8_t> v_fp8(pool, (size_t) batch * num_k_heads * head_dim_padded * kv_len_padded);
    ggml_cuda_pool_alloc<float>  v_scale(pool, (size_t) batch * num_k_heads * head_dim_padded);
    const bool smooth_v = true;
    float * v_mean_ptr = nullptr;
    std::unique_ptr<ggml_cuda_pool_alloc<float>> v_mean_storage;
    if (smooth_v) {
        v_mean_storage = std::make_unique<ggml_cuda_pool_alloc<float>>(pool, (size_t) batch * num_k_heads * head_dim_padded);
        v_mean_ptr = v_mean_storage->get();
    }

    switch (head_dim_padded) {
        case 64:
            if (smooth_v) {
                quantize_v_to_fp8<T, 64, true>(v_transposed.get(), v_fp8.get(), v_mean_ptr, v_scale.get(), FP8_SCALE_MAX, batch, num_k_heads, head_dim_padded, kv_len, stream);
            } else {
                quantize_v_to_fp8<T, 64, false>(v_transposed.get(), v_fp8.get(), nullptr, v_scale.get(), FP8_SCALE_MAX, batch, num_k_heads, head_dim_padded, kv_len, stream);
            }
            break;
        case 128:
            if (smooth_v) {
                quantize_v_to_fp8<T, 128, true>(v_transposed.get(), v_fp8.get(), v_mean_ptr, v_scale.get(), FP8_SCALE_MAX, batch, num_k_heads, head_dim_padded, kv_len, stream);
            } else {
                quantize_v_to_fp8<T, 128, false>(v_transposed.get(), v_fp8.get(), nullptr, v_scale.get(), FP8_SCALE_MAX, batch, num_k_heads, head_dim_padded, kv_len, stream);
            }
            break;
    }

    ggml_cuda_pool_alloc<T> out_padded(pool, (size_t) head_dim_padded * qo_len * num_q_heads * batch);
    const tensor_dims out_pad_dims = { head_dim_padded, qo_len, num_q_heads, batch };
    const tensor_strides out_pad_strides = make_contiguous_strides(out_pad_dims);
    const tensor_dims dst_dims = { head_dim, qo_len, num_q_heads, batch };
    const tensor_strides dst_strides = get_tensor_strides(dst);

    value_strides v_quant_strides = make_value_strides(num_k_heads, head_dim_padded, kv_len_padded);

    const float sm_scale = softmax_scale;

    switch (head_dim_padded) {
        case 64:
            if (is_causal) {
                launch_sage_kernel<T, 64, MaskMode::kCausal>(
                    q_int8.get(), k_int8.get(), v_fp8.get(), out_padded.get(),
                    q_scale.get(), k_scale.get(), v_scale.get(), v_mean_ptr,
                    batch, num_q_heads, num_k_heads, qo_len, kv_len, num_kv_groups,
                    q_pad_strides, k_pad_strides, out_pad_strides, v_quant_strides,
                    sm_scale, stream, granularity);
            } else {
                launch_sage_kernel<T, 64, MaskMode::kNone>(
                    q_int8.get(), k_int8.get(), v_fp8.get(), out_padded.get(),
                    q_scale.get(), k_scale.get(), v_scale.get(), v_mean_ptr,
                    batch, num_q_heads, num_k_heads, qo_len, kv_len, num_kv_groups,
                    q_pad_strides, k_pad_strides, out_pad_strides, v_quant_strides,
                    sm_scale, stream, granularity);
            }
            break;
        case 128:
            if (is_causal) {
                launch_sage_kernel<T, 128, MaskMode::kCausal>(
                    q_int8.get(), k_int8.get(), v_fp8.get(), out_padded.get(),
                    q_scale.get(), k_scale.get(), v_scale.get(), v_mean_ptr,
                    batch, num_q_heads, num_k_heads, qo_len, kv_len, num_kv_groups,
                    q_pad_strides, k_pad_strides, out_pad_strides, v_quant_strides,
                    sm_scale, stream, granularity);
            } else {
                launch_sage_kernel<T, 128, MaskMode::kNone>(
                    q_int8.get(), k_int8.get(), v_fp8.get(), out_padded.get(),
                    q_scale.get(), k_scale.get(), v_scale.get(), v_mean_ptr,
                    batch, num_q_heads, num_k_heads, qo_len, kv_len, num_kv_groups,
                    q_pad_strides, k_pad_strides, out_pad_strides, v_quant_strides,
                    sm_scale, stream, granularity);
            }
            break;
        default:
            GGML_ABORT("SageAttention SM89: unsupported head_dim");
    }

    trim_tensor_head_dim(
        out_padded.get(),
        head_dim_padded,
        dst_dims,
        out_pad_strides,
        dst_strides,
        static_cast<T *>(dst->data),
        stream);
}

} // namespace

void ggml_cuda_sage_attn_sm89_fp16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    switch (q->type) {
        case GGML_TYPE_F16:
            sage_attn_impl<half>(ctx, dst);
            break;
        case GGML_TYPE_BF16:
            sage_attn_impl<nv_bfloat16>(ctx, dst);
            break;
        default:
            GGML_ABORT("ggml_cuda_sage_attn_sm89_fp16: unsupported dtype");
    }
}

bool ggml_cuda_sage_attn_sm89_fp16_supported(int device, const ggml_tensor * dst) {
    const auto & dev_info = ggml_cuda_info().devices[device];
    if (dev_info.cc < GGML_CUDA_CC_ADA_LOVELACE) {
        return false;
    }

    if (dst->src[0] == nullptr || dst->src[1] == nullptr || dst->src[2] == nullptr) {
        return false;
    }

    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];

    if (!((q->type == GGML_TYPE_F16 || q->type == GGML_TYPE_BF16) &&
          q->type == k->type && q->type == v->type)) {
        return false;
    }

    const int32_t quant_gran = ggml_get_op_params_i32(dst, 3);
    if (quant_gran != GGML_SAGE_QK_GRANULARITY_PER_WARP &&
        quant_gran != GGML_SAGE_QK_GRANULARITY_PER_THREAD) {
        return false;
    }

    return true;
}
#define SAGE_DEBUG 1
