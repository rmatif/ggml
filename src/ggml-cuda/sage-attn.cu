#include "sage-attn.cuh"

#include "common.cuh"
#include "ggml-cuda.h"

#include "sage/fused_kernels.cuh"
#include "sage/qattn/qk_int_sv_f8_cuda_sm89.cuh"
#include "sage/qattn/attn_utils.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_fp8.h>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {

static void sage_debug_dump(const char * prefix, const char * tag, const void * device_ptr, size_t num_bytes, cudaStream_t stream) {
    if (prefix == nullptr || prefix[0] == '\0' || num_bytes == 0) {
        return;
    }

    std::vector<uint8_t> host(num_bytes);
    CUDA_CHECK(cudaMemcpyAsync(host.data(), device_ptr, num_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::string path = std::string(prefix) + "." + tag + ".bin";
    FILE * fp = fopen(path.c_str(), "wb");
    if (!fp) {
        fprintf(stderr, "ggml_sage_attn: failed to write dump %s\n", path.c_str());
        return;
    }
    fwrite(host.data(), 1, num_bytes, fp);
    fclose(fp);
}

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

constexpr float FP8_SCALE_MAX_FP32 = 448.0f;
constexpr float FP8_SCALE_MAX_FP16 = 2.25f;

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

template<typename T>
__global__ void permute_dshb_to_bhsd_kernel(
        const T * __restrict__ src,
        T * __restrict__ dst,
        int head_dim,
        int seq_len,
        int heads,
        int batch) {
    const int64_t total = (int64_t) batch * heads * seq_len * head_dim;
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    int64_t tmp = idx;
    const int dim = tmp % head_dim;
    tmp /= head_dim;
    const int seq = tmp % seq_len;
    tmp /= seq_len;
    const int head = tmp % heads;
    const int batch_id = tmp / heads;

    const int64_t src_idx =
        ((int64_t) dim) +
        (int64_t) head_dim * (
            seq +
            (int64_t) seq_len * (
                head +
                (int64_t) heads * batch_id));

    const int64_t dst_idx =
        (((int64_t) batch_id * heads + head) * seq_len + seq) * head_dim + dim;

    dst[dst_idx] = src[src_idx];
}

template<typename T>
void permute_dshb_to_bhsd(
        const T * src,
        T * dst,
        int head_dim,
        int seq_len,
        int heads,
        int batch,
        cudaStream_t stream) {
    const int64_t total = (int64_t) batch * heads * seq_len * head_dim;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    permute_dshb_to_bhsd_kernel<<<blocks, threads, 0, stream>>>(
        src, dst, head_dim, seq_len, heads, batch);
    CUDA_CHECK(cudaGetLastError());
}

template<typename T>
static float to_float_host(T v);

template<>
float to_float_host<half>(half v) {
    return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t *>(&v));
}

template<>
float to_float_host<nv_bfloat16>(nv_bfloat16 v) {
    return ggml_bf16_to_fp32(*reinterpret_cast<const ggml_bf16_t *>(&v));
}

inline int8_t quantize_scalar(float val, float inv_scale) {
    const float clipped = std::max(-127.0f, std::min(127.0f, std::nearbyint(val * inv_scale)));
    return (int8_t) clipped;
}

template<typename T>
void quantize_q_per_warp_host(
        const T * src_dev,
        int8_t * dst_dev,
        float * scale_dev,
        int batch,
        int heads,
        int seq_len,
        int head_dim,
        cudaStream_t stream) {
    const int warps_per_block = BLKQ / WARPQ;
    const int num_blocks = (seq_len + BLKQ - 1) / BLKQ;
    const size_t elems = (size_t) batch * heads * seq_len * head_dim;
    std::vector<T> host_in(elems);
    CUDA_CHECK(cudaMemcpyAsync(host_in.data(), src_dev, elems * sizeof(T), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<int8_t> host_out(elems, 0);
    std::vector<float> host_scale((size_t) batch * heads * num_blocks * warps_per_block, 0.0f);

    auto value_at = [&](int b, int h, int t, int d) -> float {
        const size_t idx = (((size_t) b * heads + h) * seq_len + t) * head_dim + d;
        return to_float_host(host_in[idx]);
    };

    auto out_index = [&](int b, int h, int t, int d) -> size_t {
        return (((size_t) b * heads + h) * seq_len + t) * head_dim + d;
    };

    const float eps = 1e-7f;

    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            for (int block = 0; block < num_blocks; ++block) {
                for (int warp = 0; warp < warps_per_block; ++warp) {
                    float max_abs = eps;
                    for (int token = 0; token < WARPQ; ++token) {
                        const int t = block*BLKQ + warp*WARPQ + token;
                        if (t >= seq_len) {
                            break;
                        }
                        for (int d = 0; d < head_dim; ++d) {
                            const float v = value_at(b, h, t, d);
                            max_abs = std::max(max_abs, fabsf(v));
                        }
                    }
                    const float inv_scale = 127.0f / max_abs;
                    const float scale_val = max_abs / 127.0f;
                    host_scale[(((size_t) b * heads + h) * num_blocks + block) * warps_per_block + warp] = scale_val;
                    for (int token = 0; token < WARPQ; ++token) {
                        const int t = block*BLKQ + warp*WARPQ + token;
                        if (t >= seq_len) {
                            break;
                        }
                        for (int d = 0; d < head_dim; ++d) {
                            const float v = value_at(b, h, t, d);
                            host_out[out_index(b, h, t, d)] = quantize_scalar(v, inv_scale);
                        }
                    }
                }
            }
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(dst_dev, host_out.data(), elems * sizeof(int8_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(scale_dev, host_scale.data(),
        host_scale.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<typename T>
void quantize_k_per_block_host(
        const T * src_dev,
        const T * mean_dev,
        int8_t * dst_dev,
        float * scale_dev,
        int batch,
        int heads,
        int seq_len,
        int head_dim,
        cudaStream_t stream) {
    const int num_blocks = (seq_len + BLKK - 1) / BLKK;
    const size_t elems = (size_t) batch * heads * seq_len * head_dim;
    std::vector<T> host_in(elems);
    CUDA_CHECK(cudaMemcpyAsync(host_in.data(), src_dev, elems * sizeof(T), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<int8_t> host_out(elems, 0);
    std::vector<float> host_scale((size_t) batch * heads * num_blocks, 0.0f);

    std::vector<float> host_mean;
    auto mean_at = [&](int b, int h, int d) -> float {
        if (mean_dev == nullptr) return 0.0f;
        const size_t idx = (((size_t) b * heads + h) * head_dim) + d;
        return host_mean[idx];
    };
    if (mean_dev != nullptr) {
        host_mean.resize((size_t) batch * heads * head_dim);
        CUDA_CHECK(cudaMemcpyAsync(host_mean.data(), mean_dev,
            host_mean.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    auto value_at = [&](int b, int h, int t, int d) -> float {
        const size_t idx = (((size_t) b * heads + h) * seq_len + t) * head_dim + d;
        return to_float_host(host_in[idx]);
    };
    auto out_index = [&](int b, int h, int t, int d) -> size_t {
        return (((size_t) b * heads + h) * seq_len + t) * head_dim + d;
    };

    const float eps = 1e-7f;

    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            for (int block = 0; block < num_blocks; ++block) {
                float max_abs = eps;
                for (int token = 0; token < BLKK; ++token) {
                    const int t = block*BLKK + token;
                    if (t >= seq_len) break;
                    for (int d = 0; d < head_dim; ++d) {
                        float v = value_at(b, h, t, d);
                        if (mean_dev != nullptr) {
                            v -= mean_at(b, h, d);
                        }
                        max_abs = std::max(max_abs, fabsf(v));
                    }
                }
                const float inv_scale = 127.0f / max_abs;
                const float scale_val = max_abs / 127.0f;
                host_scale[((size_t) b * heads + h) * num_blocks + block] = scale_val;
                for (int token = 0; token < BLKK; ++token) {
                    const int t = block*BLKK + token;
                    if (t >= seq_len) break;
                    for (int d = 0; d < head_dim; ++d) {
                        float v = value_at(b, h, t, d);
                        if (mean_dev != nullptr) {
                            v -= mean_at(b, h, d);
                        }
                        host_out[out_index(b, h, t, d)] = quantize_scalar(v, inv_scale);
                    }
                }
            }
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(dst_dev, host_out.data(), elems * sizeof(int8_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(scale_dev, host_scale.data(),
        host_scale.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<typename T>
void convert_dshb_to_bhsd_host(
        const T * src,
        T * dst,
        int head_dim,
        int seq_len,
        int heads,
        int batch,
        cudaStream_t stream) {
    const size_t elems = (size_t) head_dim * seq_len * heads * batch;
    std::vector<T> host_in(elems);
    std::vector<T> host_out(elems);

    CUDA_CHECK(cudaMemcpyAsync(host_in.data(), src, elems * sizeof(T), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const int64_t stride_dim = (int64_t) seq_len * heads * batch;
    const int64_t stride_seq = (int64_t) heads * batch;
    const int64_t stride_head = batch;
    const int64_t stride_batch = 1;

    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            for (int s = 0; s < seq_len; ++s) {
                for (int d = 0; d < head_dim; ++d) {
                    const size_t src_index =
                        d*stride_dim +
                        s*stride_seq +
                        h*stride_head +
                        b*stride_batch;
                    const size_t dst_index =
                        (((size_t) b * heads + h) * seq_len + s) * head_dim + d;
                    host_out[dst_index] = host_in[src_index];
                }
            }
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(dst, host_out.data(), elems * sizeof(T), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<typename T>
__global__ void convert_dshb_to_bhsd_kernel(
        const T * __restrict__ src,
        T * __restrict__ dst,
        int head_dim,
        int seq_len,
        int heads,
        int batch) {
    const size_t total = (size_t) head_dim * seq_len * heads * batch;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (size_t) blockDim.x * gridDim.x) {
        const int d = idx % head_dim;
        size_t tmp = idx / head_dim;
        const int s = tmp % seq_len;
        tmp /= seq_len;
        const int h = tmp % heads;
        const int b = tmp / heads;
        const size_t dst_index =
            (((size_t) b * heads + h) * seq_len + s) * head_dim + d;
        const size_t src_index =
            (((size_t) d * seq_len + s) * heads + h) * batch + b;
        dst[dst_index] = src[src_index];
    }
}

template<typename T>
void convert_dshb_to_bhsd_device(
        const T * src,
        T * dst,
        int head_dim,
        int seq_len,
        int heads,
        int batch,
        cudaStream_t stream) {
    const size_t total = (size_t) head_dim * seq_len * heads * batch;
    if (total == 0) {
        return;
    }
    const int threads = 256;
    const int blocks = (int) ((total + threads - 1) / threads);
    convert_dshb_to_bhsd_kernel<T><<<blocks, threads, 0, stream>>>(
        src, dst, head_dim, seq_len, heads, batch);
    CUDA_CHECK(cudaGetLastError());
}

static bool sage_force_simple_quant() {
    static bool value = getenv("GGML_SAGE_FORCE_SIMPLE_QK") != nullptr;
    return value;
}

static bool sage_force_host_quant() {
    static bool value = getenv("GGML_SAGE_FORCE_HOST_QK") != nullptr;
    return value;
}

static bool sage_force_host_permute() {
    static bool value = getenv("GGML_SAGE_FORCE_HOST_PERMUTE") != nullptr;
    return value;
}

static bool sage_force_serial_quant() {
    static bool value = getenv("GGML_SAGE_Q_SERIAL") != nullptr;
    return value;
}

static bool sage_debug_kernel() {
    static bool value = getenv("GGML_SAGE_DEBUG_KERNEL") != nullptr;
    return value;
}

static bool sage_force_v_fused() {
    static bool value = getenv("GGML_SAGE_V_FUSED") != nullptr;
    return value;
}

static bool sage_disable_v_fused() {
    static bool value = getenv("GGML_SAGE_DISABLE_V_FUSED") != nullptr;
    return value;
}

template<typename T, int HEAD_DIM>
void quantize_q_per_warp_simple(
        const T * input,
        int8_t * output,
        float * scale,
        int batch,
        int heads,
        int seq_len,
        cudaStream_t stream);

template<typename T, int HEAD_DIM, bool SUB_MEAN>
void quantize_k_per_block_simple(
        const T * input,
        const T * mean,
        int8_t * output,
        float * scale,
        int batch,
        int heads,
        int seq_len,
        cudaStream_t stream);


template<typename T, int HEAD_DIM>
void quantize_q_per_warp(
        const T * input,
        int8_t * output,
        float * scale,
        int batch,
        int heads,
        int seq_len,
        cudaStream_t stream,
        const ggml_cuda_sage::quant_debug_config & dbg = {}) {
    if (sage_force_simple_quant()) {
        fprintf(stderr, "SAGE: using simple quant for Q\n");
        quantize_q_per_warp_simple<T, HEAD_DIM>(
            input, output, scale, batch, heads, seq_len, stream);
        return;
    }
    if (sage_force_serial_quant()) {
        fprintf(stderr, "SAGE_Q_DEV: using serial quantization path\n");
    }
    constexpr int BLOCK_SIZE = BLKQ;
    constexpr int WARP_BLOCK_SIZE = WARPQ;
    constexpr int num_pack_per_thread = (WARP_BLOCK_SIZE * (HEAD_DIM / 8) + 1023) / 1024;

    const int scale_cols = ((seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE) * (BLOCK_SIZE / WARP_BLOCK_SIZE);

    const int64_t stride_seq = HEAD_DIM;
    const int64_t stride_head = HEAD_DIM * seq_len;
    const int64_t stride_batch = stride_head * heads;

    const int64_t stride_head_scale = scale_cols;
    const int64_t stride_batch_scale = heads * scale_cols;

    auto launch_kernel = [&](const T * in_ptr,
                             int8_t * out_ptr,
                             float * scale_ptr,
                             uint32_t stride_h_input_arg,
                             uint32_t stride_h_scale_arg,
                             uint32_t stride_h_output_arg,
                             uint32_t stride_bz_scale_arg,
                             uint32_t grid_heads) {
        dim3 grid(((seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE) * (BLOCK_SIZE / WARP_BLOCK_SIZE),
                  grid_heads,
                  batch);
        dim3 block(WARP_BLOCK_SIZE * (HEAD_DIM / 8) / num_pack_per_thread);

        ggml_cuda_sage::quant_int8_kernel<HEAD_DIM, WARP_BLOCK_SIZE, num_pack_per_thread, false, false, T>
            <<<grid, block, 0, stream>>>(
                const_cast<T *>(in_ptr),
                nullptr,
                out_ptr,
                scale_ptr,
                0.0f,
                seq_len,
                to_u32(stride_batch), to_u32(stride_seq), stride_h_input_arg,
                0, 0,
                to_u32(stride_batch), to_u32(stride_seq), stride_h_output_arg,
                stride_bz_scale_arg, stride_h_scale_arg,
                dbg);
        CUDA_CHECK(cudaGetLastError());
    };

    if (sage_force_serial_quant()) {
        for (int head_idx = 0; head_idx < heads; ++head_idx) {
            const size_t head_offset = (size_t) head_idx * stride_head;
            const size_t scale_offset = (size_t) head_idx * scale_cols;
            launch_kernel(
                input + head_offset,
                output + head_offset,
                scale + scale_offset,
                0u,
                0u,
                0u,
                to_u32(stride_batch_scale),
                to_u32(1));
        }
    } else {
        launch_kernel(
            input,
            output,
            scale,
            to_u32(stride_head),
            to_u32(stride_head_scale),
            to_u32(stride_head),
            to_u32(stride_batch_scale),
            to_u32(heads));
    }
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
        cudaStream_t stream,
        const ggml_cuda_sage::quant_debug_config & dbg = {}) {
    if (sage_force_simple_quant()) {
        fprintf(stderr, "SAGE: using simple quant for K\n");
        quantize_k_per_block_simple<T, HEAD_DIM, SUB_MEAN>(
            input, mean, output, scale, batch, heads, seq_len, stream);
        return;
    }
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
            to_u32(stride_batch_scale), to_u32(stride_head_scale),
            dbg);
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

template<typename T>
static void debug_compare_q_scale_host(
        const T * dev_q,
        const tensor_dims & dims,
        const float * dev_scale,
        int q_scale_cols,
        cudaStream_t stream) {
    const int warps_per_block = BLKQ / WARPQ;
    const int num_block128 = div_ceil64(dims.seq_len, BLKQ);
    const size_t tensor_elems = (size_t) dims.head_dim * dims.seq_len * dims.heads * dims.batch;
    std::vector<T> host_q(tensor_elems);
    CUDA_CHECK(cudaMemcpyAsync(host_q.data(), dev_q, tensor_elems * sizeof(T), cudaMemcpyDeviceToHost, stream));

    const size_t scale_elems = (size_t) dims.batch * dims.heads * q_scale_cols;
    std::vector<float> host_scale(scale_elems);
    CUDA_CHECK(cudaMemcpyAsync(host_scale.data(), dev_scale, scale_elems * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    fprintf(stderr, "SAGE_Q_SIMPLE host copy done\n");

    auto get_elem = [&](int b, int h, int tok, int d) -> float {
        const size_t index =
            (size_t) d +
            (size_t) tok * dims.head_dim +
            (size_t) h * (size_t) dims.head_dim * dims.seq_len +
            (size_t) b * (size_t) dims.head_dim * dims.seq_len * dims.heads;
        return ggml_cuda_sage::convert_to_float(host_q[index]);
    };

    fprintf(stderr, "SAGE_Q_SIMPLE dims: batch=%d heads=%d blocks=%d warps=%d\n",
            dims.batch, dims.heads, num_block128, warps_per_block);
    bool reported = false;
    int printed = 0;
    for (int b = 0; b < dims.batch && !reported; ++b) {
        for (int h = 0; h < dims.heads && !reported; ++h) {
            for (int blk = 0; blk < num_block128 && !reported; ++blk) {
                for (int warp = 0; warp < warps_per_block && !reported; ++warp) {
                    fprintf(stderr, "SAGE_Q_SIMPLE iter batch=%d head=%d blk=%d warp=%d\n", b, h, blk, warp);
                    const int token_start = blk * BLKQ + warp * WARPQ;
                    if (token_start >= dims.seq_len) {
                        continue;
                    }
                    const int token_end = std::min(token_start + WARPQ, (int) dims.seq_len);
                    float max_val = 1e-7f;
                    for (int tok = token_start; tok < token_end; ++tok) {
                        for (int d = 0; d < dims.head_dim; ++d) {
                            max_val = std::max(max_val, fabsf(get_elem(b, h, tok, d)));
                        }
                    }
                    const float expected_scale = max_val / 127.0f;
                    const size_t scale_index =
                        (size_t) b * dims.heads * q_scale_cols +
                        (size_t) h * q_scale_cols +
                        (size_t) blk * warps_per_block + warp;
                    const float recorded = host_scale[scale_index];
                    if (printed < 4) {
                        fprintf(stderr,
                                "SAGE_Q_SIMPLE block=%d warp=%d token_start=%d expected=%g recorded=%g\n",
                                blk, warp, token_start, expected_scale, recorded);
                        printed++;
                    }
                    if (fabsf(expected_scale - recorded) > 1e-4f) {
                        fprintf(stderr,
                                "SAGE_Q_SIMPLE mismatch batch=%d head=%d blk=%d warp=%d expected=%g recorded=%g\n",
                                b, h, blk, warp, expected_scale, recorded);
                        reported = true;
                    }
                }
            }
        }
    }
    fprintf(stderr, "SAGE_Q_SIMPLE host check done printed=%d reported=%d\n", printed, reported ? 1 : 0);
}

template<typename T>
static void debug_compare_k_scale_host(
        const T * dev_k,
        const tensor_dims & dims,
        const float * dev_scale,
        int k_scale_cols,
        bool sub_mean,
        const T * dev_mean,
        cudaStream_t stream) {
    const int num_blocks = div_ceil64(dims.seq_len, BLKK);
    fprintf(stderr, "SAGE_K_SIMPLE dims: batch=%d heads=%d blocks=%d\n",
            dims.batch, dims.heads, num_blocks);
    const size_t tensor_elems = (size_t) dims.head_dim * dims.seq_len * dims.heads * dims.batch;
    std::vector<T> host_k(tensor_elems);
    CUDA_CHECK(cudaMemcpyAsync(host_k.data(), dev_k, tensor_elems * sizeof(T), cudaMemcpyDeviceToHost, stream));

    std::vector<float> host_scale((size_t) dims.batch * dims.heads * k_scale_cols);
    CUDA_CHECK(cudaMemcpyAsync(host_scale.data(), dev_scale, host_scale.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));

    std::vector<T> host_mean;
    if (sub_mean) {
        const size_t mean_elems = (size_t) dims.head_dim * dims.heads * dims.batch;
        host_mean.resize(mean_elems);
        CUDA_CHECK(cudaMemcpyAsync(host_mean.data(), dev_mean, mean_elems * sizeof(T), cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto get_elem = [&](int b, int h, int tok, int d) -> float {
        const size_t index =
            (size_t) d +
            (size_t) tok * dims.head_dim +
            (size_t) h * (size_t) dims.head_dim * dims.seq_len +
            (size_t) b * (size_t) dims.head_dim * dims.seq_len * dims.heads;
        return ggml_cuda_sage::convert_to_float(host_k[index]);
    };
    auto get_mean = [&](int b, int h, int d) -> float {
        if (!sub_mean) {
            return 0.0f;
        }
        const size_t index =
            (size_t) d +
            (size_t) h * (size_t) dims.head_dim +
            (size_t) b * (size_t) dims.head_dim * dims.heads;
        return ggml_cuda_sage::convert_to_float(host_mean[index]);
    };

    bool reported = false;
    int printed = 0;
    for (int b = 0; b < dims.batch && !reported; ++b) {
        for (int h = 0; h < dims.heads && !reported; ++h) {
            for (int blk = 0; blk < num_blocks && !reported; ++blk) {
                const int token_start = blk * BLKK;
                if (token_start >= dims.seq_len) {
                    continue;
                }
                const int token_end = std::min(token_start + BLKK, (int) dims.seq_len);
                float max_val = 1e-7f;
                const float mean_val = get_mean(b, h, 0); // mean stored per dim; subtract inside loop
                for (int tok = token_start; tok < token_end; ++tok) {
                    for (int d = 0; d < dims.head_dim; ++d) {
                        float val = get_elem(b, h, tok, d);
                        if (sub_mean) {
                            val -= get_mean(b, h, d);
                        }
                        max_val = std::max(max_val, fabsf(val));
                    }
                }
                const float expected_scale = max_val / 127.0f;
                const size_t scale_index =
                    (size_t) b * dims.heads * k_scale_cols +
                    (size_t) h * k_scale_cols +
                    blk;
                const float recorded = host_scale[scale_index];
                if (printed < 4) {
                    fprintf(stderr,
                            "SAGE_K_SIMPLE block=%d token_start=%d expected=%g recorded=%g\n",
                            blk, token_start, expected_scale, recorded);
                    printed++;
                }
                if (fabsf(expected_scale - recorded) > 1e-4f) {
                    fprintf(stderr,
                            "SAGE_K_SIMPLE mismatch batch=%d head=%d blk=%d expected=%g recorded=%g\n",
                            b, h, blk, expected_scale, recorded);
                    reported = true;
                }
            }
        }
    }
    fprintf(stderr, "SAGE_K_SIMPLE host check done printed=%d reported=%d\n", printed, reported ? 1 : 0);
}

template<typename T, int HEAD_DIM>
__global__ void quantize_q_per_warp_simple_kernel(
        const T * __restrict__ input,
        int8_t * __restrict__ output,
        float * __restrict__ scale,
        int seq_len,
        uint32_t stride_batch,
        uint32_t stride_seq,
        uint32_t stride_head,
        uint32_t stride_batch_scale,
        uint32_t stride_head_scale) {
    constexpr int WARPS_PER_BLOCK = BLKQ / WARPQ;
    const int block128 = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int batch_idx = blockIdx.z;

    __shared__ float s_block_max[WARPS_PER_BLOCK];

    for (int warp_local = 0; warp_local < WARPS_PER_BLOCK; ++warp_local) {
        const int token_start = block128 * BLKQ + warp_local * WARPQ;
        if (token_start >= seq_len) {
            continue;
        }
        const int token_end = min(token_start + WARPQ, seq_len);

        float local_max = 1e-7f;
        for (int tok = token_start; tok < token_end; ++tok) {
            const T * src = input +
                (size_t) batch_idx * stride_batch +
                (size_t) head_idx  * stride_head +
                (size_t) tok       * stride_seq;
            for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
                const float val = ggml_cuda_sage::convert_to_float(src[d]);
                local_max = fmaxf(local_max, fabsf(val));
            }
        }

        if (threadIdx.x == 0) {
            s_block_max[warp_local] = 1e-7f;
        }
        __syncthreads();
        atomic_max_relaxed(&s_block_max[warp_local], local_max);
        __syncthreads();

        const float max_val = s_block_max[warp_local];
        if (threadIdx.x == 0) {
            const size_t scale_index =
                (size_t) batch_idx * stride_batch_scale +
                (size_t) head_idx  * stride_head_scale +
                block128 * WARPS_PER_BLOCK + warp_local;
            scale[scale_index] = max_val / 127.0f;
        }

        const float inv_scale = (max_val > 0.0f) ? (127.0f / max_val) : 0.0f;

        for (int tok = token_start; tok < token_end; ++tok) {
            const T * src = input +
                (size_t) batch_idx * stride_batch +
                (size_t) head_idx  * stride_head +
                (size_t) tok       * stride_seq;
            int8_t * dst = output +
                (size_t) batch_idx * stride_batch +
                (size_t) head_idx  * stride_head +
                (size_t) tok       * stride_seq;
            for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
                const float val = ggml_cuda_sage::convert_to_float(src[d]) * inv_scale;
                dst[d] = float_to_int8_rn(val);
            }
        }
        __syncthreads();
    }
}

template<typename T, int HEAD_DIM, bool SUB_MEAN>
__global__ void quantize_k_per_block_simple_kernel(
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
    constexpr int TOKENS_PER_BLOCK = BLKK;
    const int block_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int batch_idx = blockIdx.z;

    const int token_start = block_idx * TOKENS_PER_BLOCK;
    const int token_end = min(token_start + TOKENS_PER_BLOCK, seq_len);

    float local_max = 1e-7f;
    for (int tok = token_start; tok < token_end; ++tok) {
        const T * src = input +
            (size_t) batch_idx * stride_batch +
            (size_t) head_idx  * stride_head +
            (size_t) tok       * stride_seq;
        for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
            float val = ggml_cuda_sage::convert_to_float(src[d]);
            if constexpr (SUB_MEAN) {
                const T * mean_src = mean +
                    (size_t) batch_idx * stride_batch_mean +
                    (size_t) head_idx  * stride_head_mean +
                    d;
                val -= ggml_cuda_sage::convert_to_float(mean_src[0]);
            }
            local_max = fmaxf(local_max, fabsf(val));
        }
    }

    __shared__ float s_block_max;
    if (threadIdx.x == 0) {
        s_block_max = 1e-7f;
    }
    __syncthreads();
    atomic_max_relaxed(&s_block_max, local_max);
    __syncthreads();

    const float max_val = s_block_max;
    if (threadIdx.x == 0) {
        const size_t scale_index =
            (size_t) batch_idx * stride_batch_scale +
            (size_t) head_idx  * stride_head_scale +
            block_idx;
        scale[scale_index] = max_val / 127.0f;
    }

    const float inv_scale = (max_val > 0.0f) ? (127.0f / max_val) : 0.0f;

    for (int tok = token_start; tok < token_end; ++tok) {
        const T * src = input +
            (size_t) batch_idx * stride_batch +
            (size_t) head_idx  * stride_head +
            (size_t) tok       * stride_seq;
        int8_t * dst = output +
            (size_t) batch_idx * stride_batch +
            (size_t) head_idx  * stride_head +
            (size_t) tok       * stride_seq;
        for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
            float val = ggml_cuda_sage::convert_to_float(src[d]);
            if constexpr (SUB_MEAN) {
                const T * mean_src = mean +
                    (size_t) batch_idx * stride_batch_mean +
                    (size_t) head_idx  * stride_head_mean +
                    d;
                val -= ggml_cuda_sage::convert_to_float(mean_src[0]);
            }
            const float q = val * inv_scale;
            dst[d] = float_to_int8_rn(q);
        }
    }
}

template<typename T, int HEAD_DIM>
__global__ void compute_q_block_amax_kernel(
        const T * __restrict__ input,
        float * __restrict__ amax,
        int seq_len,
        uint32_t stride_batch,
        uint32_t stride_seq,
        uint32_t stride_head,
        int scale_cols) {
    constexpr int BLOCK_SIZE = BLKQ;
    constexpr int WARP_BLOCK_SIZE = WARPQ;
    const uint32_t bx = blockIdx.x;
    const uint32_t head_id = blockIdx.y;
    const uint32_t batch_id = blockIdx.z;

    const uint32_t block128 = bx / (BLOCK_SIZE / WARP_BLOCK_SIZE);
    const uint32_t offset = bx % (BLOCK_SIZE / WARP_BLOCK_SIZE);
    const int token_start = block128 * BLOCK_SIZE + offset * WARP_BLOCK_SIZE;
    const int token_end = min(token_start + WARP_BLOCK_SIZE, seq_len);

    float local_max = 1e-7f;
    if (token_start < seq_len) {
        for (int tok = token_start; tok < token_end; ++tok) {
            const T * tok_ptr = input +
                (size_t) batch_id * stride_batch +
                (size_t) head_id * stride_head +
                (size_t) tok * stride_seq;
            for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
                local_max = fmaxf(local_max, fabsf(ggml_cuda_sage::convert_to_float(tok_ptr[d])));
            }
        }
    }

    const float block_max = vllm::blockReduceMax(local_max);
    if (threadIdx.x == 0) {
        const uint32_t idx = batch_id * (gridDim.y * scale_cols) + head_id * scale_cols + bx;
        amax[idx] = block_max;
    }
}

template<typename T, int HEAD_DIM>
__global__ void compute_k_block_amax_kernel(
        const T * __restrict__ input,
        float * __restrict__ amax,
        int seq_len,
        uint32_t stride_batch,
        uint32_t stride_seq,
        uint32_t stride_head,
        int scale_cols) {
    constexpr int BLOCK_SIZE = BLKK;
    const uint32_t bx = blockIdx.x;
    const uint32_t head_id = blockIdx.y;
    const uint32_t batch_id = blockIdx.z;

    const int token_start = bx * BLOCK_SIZE;
    const int token_end = min(token_start + BLOCK_SIZE, seq_len);

    float local_max = 1e-7f;
    if (token_start < seq_len) {
        for (int tok = token_start; tok < token_end; ++tok) {
            const T * tok_ptr = input +
                (size_t) batch_id * stride_batch +
                (size_t) head_id * stride_head +
                (size_t) tok * stride_seq;
            for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
                local_max = fmaxf(local_max, fabsf(ggml_cuda_sage::convert_to_float(tok_ptr[d])));
            }
        }
    }

    const float block_max = vllm::blockReduceMax(local_max);
    if (threadIdx.x == 0) {
        const uint32_t idx = batch_id * (gridDim.y * scale_cols) + head_id * scale_cols + bx;
        amax[idx] = block_max;
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
void quantize_q_per_warp_simple(
        const T * input,
        int8_t * output,
        float * scale,
        int batch,
        int heads,
        int seq_len,
        cudaStream_t stream) {
    const int warps_per_block = BLKQ / WARPQ;
    const int num_block128 = (seq_len + BLKQ - 1) / BLKQ;
    const int scale_cols = num_block128 * warps_per_block;

    const int64_t stride_seq = HEAD_DIM;
    const int64_t stride_head = HEAD_DIM * seq_len;
    const int64_t stride_batch = stride_head * heads;
    const int64_t stride_head_scale = scale_cols;
    const int64_t stride_batch_scale = heads * scale_cols;

    dim3 grid(num_block128, heads, batch);
    dim3 block(128);

    quantize_q_per_warp_simple_kernel<T, HEAD_DIM><<<grid, block, 0, stream>>>(
        input,
        output,
        scale,
        seq_len,
        to_u32(stride_batch),
        to_u32(stride_seq),
        to_u32(stride_head),
        to_u32(stride_batch_scale),
        to_u32(stride_head_scale));
    CUDA_CHECK(cudaGetLastError());
}

template<typename T, int HEAD_DIM, bool SUB_MEAN>
void quantize_k_per_block_simple(
        const T * input,
        const T * mean,
        int8_t * output,
        float * scale,
        int batch,
        int heads,
        int seq_len,
        cudaStream_t stream) {
    const int scale_cols = (seq_len + BLKK - 1) / BLKK;

    const int64_t stride_seq = HEAD_DIM;
    const int64_t stride_head = HEAD_DIM * seq_len;
    const int64_t stride_batch = stride_head * heads;
    const int64_t stride_head_scale = scale_cols;
    const int64_t stride_batch_scale = heads * scale_cols;
    const int64_t stride_head_mean = SUB_MEAN ? HEAD_DIM : 0;
    const int64_t stride_batch_mean = SUB_MEAN ? HEAD_DIM * heads : 0;

    dim3 grid(scale_cols, heads, batch);
    dim3 block(128);

    quantize_k_per_block_simple_kernel<T, HEAD_DIM, SUB_MEAN><<<grid, block, 0, stream>>>(
        input,
        mean,
        output,
        scale,
        seq_len,
        to_u32(stride_batch),
        to_u32(stride_seq),
        to_u32(stride_head),
        SUB_MEAN ? to_u32(stride_batch_mean) : 0,
        SUB_MEAN ? to_u32(stride_head_mean) : 0,
        to_u32(stride_batch_scale),
        to_u32(stride_head_scale));
    CUDA_CHECK(cudaGetLastError());
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

__device__ __forceinline__ int sage_fp8_permute_token(int token) {
    const int low = token & 0xF;
    const int high = token & ~0xF;
    const int b0 = low & 1;
    const int b1 = (low >> 1) & 1;
    const int b2 = (low >> 2) & 1;
    const int b3 = (low >> 3) & 1;
    const int new_low = b0 | (b3 << 1) | (b1 << 2) | (b2 << 3);
    return high | new_low;
}

static inline int sage_fp8_permute_token_host(int token) {
    const int low = token & 0xF;
    const int high = token & ~0xF;
    const int b0 = low & 1;
    const int b1 = (low >> 1) & 1;
    const int b2 = (low >> 2) & 1;
    const int b3 = (low >> 3) & 1;
    const int new_low = b0 | (b3 << 1) | (b1 << 2) | (b2 << 3);
    return high | new_low;
}

template<typename T, int HEAD_DIM>
void transpose_pad_permute_host(
        const T * input,
        T * output,
        int batch,
        int heads,
        int seq_len,
        int padded_seq_len,
        cudaStream_t stream) {
    const int padded = ((seq_len + CTA_TRANSPOSE - 1) / CTA_TRANSPOSE) * CTA_TRANSPOSE;
    GGML_ASSERT(padded == padded_seq_len);

    const size_t in_elems = (size_t) HEAD_DIM * seq_len * heads * batch;
    const size_t out_elems = (size_t) HEAD_DIM * padded_seq_len * heads * batch;

    std::vector<T> host_in(in_elems);
    std::vector<T> host_out(out_elems, T(0));

    CUDA_CHECK(cudaMemcpyAsync(host_in.data(), input, in_elems * sizeof(T), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            for (int d = 0; d < HEAD_DIM; ++d) {
                for (int token = 0; token < padded_seq_len; ++token) {
                    T val = T(0);
                    if (token < seq_len) {
                        const size_t src_index =
                            d +
                            (size_t) HEAD_DIM * (
                                token +
                                (size_t) seq_len * (
                                    h +
                                    (size_t) heads * b));
                        val = host_in[src_index];
                    }
                    const int permuted = sage_fp8_permute_token_host(token);
                    const size_t dst_index =
                        (((size_t) b * heads + h) * HEAD_DIM + d) * padded_seq_len + permuted;
                    host_out[dst_index] = val;
                }
            }
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(output, host_out.data(), out_elems * sizeof(T), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<typename T, int HEAD_DIM>
__global__ void transpose_pad_permute_strided_kernel(
        const T * __restrict__ src,
        T * __restrict__ dst,
        uint32_t stride_batch,
        uint32_t stride_head,
        uint32_t stride_seq,
        int head_dim_src,
        int seq_len,
        int padded_seq_len,
        int heads,
        int batch) {
    constexpr uint32_t CTA_SIZE = CTA_TRANSPOSE;
    constexpr uint32_t pack_size = 8;
    const uint32_t num_threads_per_token = HEAD_DIM / pack_size;
    const uint32_t num_threads_per_cta = CTA_SIZE / pack_size;

    const uint32_t bx = blockIdx.x;
    const uint32_t head_id = blockIdx.y;
    const uint32_t batch_id = blockIdx.z;
    const uint32_t thread_id = threadIdx.x;

    const uint32_t token_base = bx * CTA_SIZE;
    const uint32_t local_token = thread_id / num_threads_per_token;
    const uint32_t token = token_base + local_token;
    const uint32_t dim_lane = (thread_id % num_threads_per_token) * pack_size;

    const T * input_ptr_base = src +
        (size_t) batch_id * stride_batch +
        (size_t) head_id * stride_head +
        (size_t) token * stride_seq +
        dim_lane;

    T * output_ptr_base = dst +
        (((size_t) batch_id * heads + head_id) * (size_t) padded_seq_len * HEAD_DIM) +
        dim_lane;

    __shared__ T shared_load[CTA_SIZE][HEAD_DIM];
    __shared__ T shared_store[HEAD_DIM][CTA_SIZE];

    const uint32_t smem_load_row_base = ((thread_id/num_threads_per_token)/16)*16;
    const uint32_t smem_load_row_mod = (thread_id/num_threads_per_token) % 16;
    const uint32_t smem_load_row = smem_load_row_base
        + (smem_load_row_mod/8)*2
        + ((smem_load_row_mod/2) % 4)*4
        + (smem_load_row_mod % 2);

    const bool pred = (token < (uint32_t) seq_len) && (dim_lane < (uint32_t) head_dim_src);

    cp_async::pred_load_128b<cp_async::PrefetchMode::kNoPrefetch, cp_async::SharedMemFillMode::kFillZero>(
        shared_load[smem_load_row] + (thread_id % num_threads_per_token)*pack_size,
        input_ptr_base,
        pred);
    cp_async::commit_group();
    cp_async::wait_group<0>();
    __syncthreads();

    const uint32_t smem_row_base = thread_id % CTA_SIZE;
    const uint32_t smem_col_base = thread_id / CTA_SIZE;
    const uint32_t smem_col_stride = HEAD_DIM / 8;

#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
        shared_store[smem_col_base + i*smem_col_stride][smem_row_base] =
            shared_load[smem_row_base][smem_col_base + i*smem_col_stride];
    }
    __syncthreads();

    const uint32_t permuted = sage_fp8_permute_token(token);
    T * out_ptr = output_ptr_base + (size_t) permuted * HEAD_DIM;
    *(float4*)(out_ptr) =
        *(float4*)(&shared_store[thread_id/num_threads_per_cta][(thread_id % num_threads_per_cta)*pack_size]);
}

template<typename T, int HEAD_DIM>
void transpose_pad_permute_strided(
        const T * input,
        const tensor_strides & strides,
        int head_dim_src,
        int batch,
        int heads,
        int seq_len,
        int padded_seq_len,
        T * output,
        cudaStream_t stream) {
    if (batch == 0 || heads == 0 || padded_seq_len == 0) {
        return;
    }
    const int threads = CTA_TRANSPOSE * (HEAD_DIM / 8);
    const int blocks_x = (int) div_ceil64(padded_seq_len, CTA_TRANSPOSE);
    dim3 grid(blocks_x, heads, batch);
    dim3 block(threads);
    transpose_pad_permute_strided_kernel<T, HEAD_DIM><<<grid, block, 0, stream>>>(
        input,
        output,
        to_u32(strides.stride_batch),
        to_u32(strides.stride_head),
        to_u32(strides.stride_seq),
        head_dim_src,
        seq_len,
        padded_seq_len,
        heads,
        batch);
    CUDA_CHECK(cudaGetLastError());
}

template<typename T, bool SUB_MEAN>
void quantize_v_per_channel(
        T * input,
        int8_t * output,
        float * mean,
        float * scale,
        float scale_max,
        int batch,
        int heads,
        int head_dim,
        int seq_len,
        int seq_len_padded,
        cudaStream_t stream) {
    const int64_t stride_dim = seq_len_padded;
    const int64_t stride_head = head_dim * stride_dim;
    const int64_t stride_batch = heads * stride_head;

    const int64_t stride_batch_scale = heads * head_dim;
    const int64_t stride_head_scale = head_dim;

    dim3 grid(heads, batch, head_dim);
    dim3 block(256);

    ggml_cuda_sage::mean_scale_kernel<64, SUB_MEAN, T><<<grid, block, 0, stream>>>(
        input,
        output,
        SUB_MEAN ? mean : nullptr,
        scale,
        scale_max,
        seq_len,
        to_u32(stride_batch), to_u32(stride_dim), to_u32(stride_head),
        to_u32(stride_batch), to_u32(stride_dim), to_u32(stride_head),
        SUB_MEAN ? to_u32(stride_batch_scale) : 0,
        SUB_MEAN ? to_u32(stride_head_scale) : 0,
        to_u32(stride_batch_scale), to_u32(stride_head_scale));
    CUDA_CHECK(cudaGetLastError());
}

template<typename T, int HEAD_DIM, bool SUB_MEAN, bool INPUT_PERMUTED>
__global__ void quantize_v_fused_kernel(
        const T * __restrict__ input,
        int8_t * __restrict__ output,
        float * __restrict__ mean,
        float * __restrict__ scale,
        const float scale_max,
        const uint32_t seq_len,
        const uint32_t seq_len_padded,
        const uint32_t stride_bz_input,
        const uint32_t stride_head_input,
        const uint32_t stride_seq_input,
        const uint32_t stride_bz_output,
        const uint32_t stride_head_output,
        const uint32_t stride_d_output,
        const uint32_t stride_bz_mean,
        const uint32_t stride_head_mean,
        const uint32_t stride_bz_scale,
        const uint32_t stride_head_scale,
        const uint32_t head_dim_src) {
    const uint32_t head_id = blockIdx.x;
    const uint32_t batch_id = blockIdx.y;
    const uint32_t d_id = blockIdx.z;
    const uint32_t lane = threadIdx.x;
    const uint32_t num_threads = blockDim.x;

    const bool valid_dim = d_id < head_dim_src;
    const T * input_base = valid_dim ? (input +
        (size_t) batch_id * stride_bz_input +
        (size_t) head_id * stride_head_input +
        d_id) : nullptr;

    int8_t * output_base = output +
        (size_t) batch_id * stride_bz_output +
        (size_t) head_id * stride_head_output +
        (size_t) d_id * stride_d_output;

    __shared__ float shared_mean;
    __shared__ float shared_amax;

    if (!valid_dim) {
        if (lane == 0) {
            shared_mean = 0.0f;
            shared_amax = 1.0f;
            if (mean != nullptr) {
                mean[batch_id*stride_bz_mean + head_id*stride_head_mean + d_id] = 0.0f;
            }
            if (scale != nullptr) {
                scale[batch_id*stride_bz_scale + head_id*stride_head_scale + d_id] = 1.0f / scale_max;
            }
        }
        __syncthreads();
        for (uint32_t token = lane; token < seq_len_padded; token += num_threads) {
            const int permuted = INPUT_PERMUTED ? token : sage_fp8_permute_token((int) token);
            output_base[permuted] = 0;
        }
        return;
    }

    float local_max = -1e6f;
    float local_min = 1e6f;
    float local_sum = 0.0f;

    for (uint32_t token = lane; token < seq_len; token += num_threads) {
        const T * ptr = input_base + (size_t) token * stride_seq_input;
        const float v = ggml_cuda_sage::convert_to_float(*ptr);
        local_max = fmaxf(local_max, v);
        local_min = fminf(local_min, v);
        if constexpr (SUB_MEAN) {
            local_sum += v;
        }
    }

    const float block_max = vllm::blockReduceMax(local_max);
    const float block_min = vllm::blockReduceMin(local_min);
    float block_sum = 0.0f;
    if constexpr (SUB_MEAN) {
        block_sum = vllm::blockReduceSum(local_sum);
    }

    if (lane == 0) {
        float amax = fmaxf(fabsf(block_max), fabsf(block_min));
        float mean_val = 0.0f;
        if constexpr (SUB_MEAN) {
            mean_val = (seq_len > 0) ? (block_sum / seq_len) : 0.0f;
            if (mean != nullptr) {
                mean[batch_id*stride_bz_mean + head_id*stride_head_mean + d_id] = mean_val;
            }
            amax = fmaxf(fabsf(block_max - mean_val), fabsf(block_min - mean_val));
        } else if (mean != nullptr) {
            mean[batch_id*stride_bz_mean + head_id*stride_head_mean + d_id] = 0.0f;
        }
        shared_mean = mean_val;
        if (amax <= 0.0f) {
            amax = 1.0f;
        }
        shared_amax = amax;
        if (scale != nullptr) {
            scale[batch_id*stride_bz_scale + head_id*stride_head_scale + d_id] = amax / scale_max;
        }
    }

    __syncthreads();

    const float mean_val = shared_mean;
    const float inv_scale = scale_max / shared_amax;

    for (uint32_t token = lane; token < seq_len_padded; token += num_threads) {
        float v = 0.0f;
        if (token < seq_len) {
            const T * ptr = input_base + (size_t) token * stride_seq_input;
            v = ggml_cuda_sage::convert_to_float(*ptr);
            if constexpr (SUB_MEAN) {
                v -= mean_val;
            }
            v *= inv_scale;
        }
        const uint32_t dst_token = INPUT_PERMUTED ? token : (uint32_t) sage_fp8_permute_token((int) token);
        const __nv_fp8_storage_t fp8 =
            __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
        output_base[dst_token] = static_cast<int8_t>(fp8);
    }
}

template<typename T, int HEAD_DIM, bool SUB_MEAN>
void quantize_v_per_channel_fused(
        const T * input,
        int8_t * output,
        float * mean,
        float * scale,
        float scale_max,
        int batch,
        int heads,
        int head_dim_src,
        int seq_len,
        int seq_len_padded,
        const tensor_strides & input_strides,
        const value_strides & output_strides,
        uint32_t stride_bz_mean,
        uint32_t stride_head_mean,
        uint32_t stride_bz_scale,
        uint32_t stride_head_scale,
        bool input_permuted,
        cudaStream_t stream) {
    dim3 grid(heads, batch, HEAD_DIM);
    dim3 block(256);
    if (input_permuted) {
        quantize_v_fused_kernel<T, HEAD_DIM, SUB_MEAN, true><<<grid, block, 0, stream>>>(
            input,
            output,
            mean,
            scale,
            scale_max,
            (uint32_t) seq_len,
            (uint32_t) seq_len_padded,
            to_u32(input_strides.stride_batch),
            to_u32(input_strides.stride_head),
            to_u32(input_strides.stride_seq),
            output_strides.stride_bz,
            output_strides.stride_h,
            output_strides.stride_d,
            stride_bz_mean,
            stride_head_mean,
            stride_bz_scale,
            stride_head_scale,
            (uint32_t) head_dim_src);
    } else {
        quantize_v_fused_kernel<T, HEAD_DIM, SUB_MEAN, false><<<grid, block, 0, stream>>>(
            input,
            output,
            mean,
            scale,
            scale_max,
            (uint32_t) seq_len,
            (uint32_t) seq_len_padded,
            to_u32(input_strides.stride_batch),
            to_u32(input_strides.stride_head),
            to_u32(input_strides.stride_seq),
            output_strides.stride_bz,
            output_strides.stride_h,
            output_strides.stride_d,
            stride_bz_mean,
            stride_head_mean,
            stride_bz_scale,
            stride_head_scale,
            (uint32_t) head_dim_src);
    }
    CUDA_CHECK(cudaGetLastError());
}

template<typename T, int HEAD_DIM, MaskMode MODE, QuantGranularity Q_GRAN, QuantGranularity K_GRAN,
         typename SVAccum, bool USE_INST_BUFFER, bool FUSE_V_MEAN, bool USE_PV_FP16_ACCUM>
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
        cudaStream_t stream,
        const sage_kernel_debug_q & dbg_q = {}) {
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
            SVAccum,
            USE_INST_BUFFER,
            T,
            ComputeUnit::kCudaCore,
            MODE,
            false,
            true,
            FUSE_V_MEAN,
            USE_PV_FP16_ACCUM>;

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
        sm_scale,
        dbg_q);
    CUDA_CHECK(cudaGetLastError());
}

template<typename T, int HEAD_DIM, MaskMode MODE, typename SVAccum, bool USE_INST_BUFFER, bool USE_PV_FP16_ACCUM>
void launch_sage_kernel_dispatch(
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
        ggml_sage_qk_granularity granularity,
        bool fuse_v_mean,
        const sage_kernel_debug_q & dbg_q = {}) {
    GGML_ASSERT(granularity == GGML_SAGE_QK_GRANULARITY_PER_WARP);
    if (fuse_v_mean) {
        launch_sage_kernel_variant<T, HEAD_DIM, MODE, QuantGranularity::kPerWarp, QuantGranularity::kPerBlock,
                                   SVAccum, USE_INST_BUFFER, true, USE_PV_FP16_ACCUM>(
            q, k, v, output, q_scale, k_scale, v_scale, v_mean,
            batch, num_q_heads, num_k_heads, qo_len, kv_len, num_kv_groups,
            q_strides, k_strides, o_strides, v_strides, sm_scale, stream, dbg_q);
    } else {
        launch_sage_kernel_variant<T, HEAD_DIM, MODE, QuantGranularity::kPerWarp, QuantGranularity::kPerBlock,
                                   SVAccum, USE_INST_BUFFER, false, USE_PV_FP16_ACCUM>(
            q, k, v, output, q_scale, k_scale, v_scale, nullptr,
            batch, num_q_heads, num_k_heads, qo_len, kv_len, num_kv_groups,
            q_strides, k_strides, o_strides, v_strides, sm_scale, stream, dbg_q);
    }
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
        ggml_sage_qk_granularity granularity,
        ggml_sage_pv_accum pv_accum,
        const sage_kernel_debug_q & dbg_q = {}) {
    const bool fuse_v_mean = v_mean != nullptr;

    switch (pv_accum) {
        case GGML_SAGE_PV_ACCUM_FP32:
            launch_sage_kernel_dispatch<T, HEAD_DIM, MODE, float, false, false>(
                q, k, v, output, q_scale, k_scale, v_scale, v_mean,
                batch, num_q_heads, num_k_heads, qo_len, kv_len, num_kv_groups,
                q_strides, k_strides, o_strides, v_strides,
                sm_scale, stream, granularity, fuse_v_mean, dbg_q);
            break;
        case GGML_SAGE_PV_ACCUM_FP32_FP32:
            launch_sage_kernel_dispatch<T, HEAD_DIM, MODE, float, true, false>(
                q, k, v, output, q_scale, k_scale, v_scale, nullptr,
                batch, num_q_heads, num_k_heads, qo_len, kv_len, num_kv_groups,
                q_strides, k_strides, o_strides, v_strides,
                sm_scale, stream, granularity, false, dbg_q);
            break;
        case GGML_SAGE_PV_ACCUM_FP32_FP16:
            launch_sage_kernel_dispatch<T, HEAD_DIM, MODE, float, true, true>(
                q, k, v, output, q_scale, k_scale, v_scale, nullptr,
                batch, num_q_heads, num_k_heads, qo_len, kv_len, num_kv_groups,
                q_strides, k_strides, o_strides, v_strides,
                sm_scale, stream, granularity, false, dbg_q);
            break;
        default:
            GGML_ABORT("SageAttention SM89: unsupported pv_accum");
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
    const char * dump_prefix = getenv("GGML_SAGE_DUMP");
    const bool debug_quant = getenv("GGML_SAGE_DEBUG_QUANT") != nullptr;
    const bool debug_quant_host = getenv("GGML_SAGE_DEBUG_QUANT_HOST") != nullptr;
    const bool debug_quant_dev = getenv("GGML_SAGE_DEBUG_QUANT_DEV") != nullptr;
    if (debug_quant_dev) {
        fprintf(stderr, "SAGE_DEBUG: quant_dev enabled\n");
    }
    const bool debug_kernel = sage_debug_kernel();
    const bool debug_kernel_iters = getenv("GGML_SAGE_DEBUG_KERNEL_ITERS") != nullptr;
    const bool dump_int8_qk = getenv("GGML_SAGE_DUMP_QK_INT8") != nullptr;
    const bool force_host_permute = sage_force_host_permute();
    const bool debug_v_compare = getenv("GGML_SAGE_DEBUG_V_FUSED") != nullptr;

    GGML_ASSERT(dst->src[0] && dst->src[1] && dst->src[2]);

    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];

    const float softmax_scale = ggml_get_op_params_f32(dst, 0);
    const bool  is_causal     = ggml_get_op_params_i32(dst, 1) != 0;
    const bool  smooth_k      = ggml_get_op_params_i32(dst, 2) != 0;
    const bool  smooth_v_param = ggml_get_op_params_i32(dst, 3) != 0;
    const auto  pv_accum      = static_cast<ggml_sage_pv_accum>(ggml_get_op_params_i32(dst, 4));
    auto        granularity   = static_cast<ggml_sage_qk_granularity>(ggml_get_op_params_i32(dst, 5));
    bool        per_thread_qk = (granularity == GGML_SAGE_QK_GRANULARITY_PER_THREAD);
    if (per_thread_qk) {
        GGML_LOG_WARN("SageAttention CUDA: per-thread Q/K granularity not supported upstream, falling back to per-warp.\n");
        granularity = GGML_SAGE_QK_GRANULARITY_PER_WARP;
        per_thread_qk = false;
    }

    GGML_UNUSED(is_causal);

    const tensor_dims q_dims = get_tensor_dims(q);
    const tensor_dims k_dims = get_tensor_dims(k);
    const tensor_dims v_dims = get_tensor_dims(v);

    GGML_ASSERT(q_dims.head_dim == k_dims.head_dim);
    GGML_ASSERT(q_dims.head_dim == v_dims.head_dim);

    const int head_dim = q_dims.head_dim;
    const int v_head_dim = v_dims.head_dim;
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
    const tensor_dims v_pad_dims = { head_dim_padded, kv_len, v_dims.heads, v_dims.batch };

    const tensor_strides q_strides = get_tensor_strides(q);
    const tensor_strides k_strides = get_tensor_strides(k);
    const tensor_strides v_strides = get_tensor_strides(v);

    const tensor_strides q_pad_strides = make_contiguous_strides(q_pad_dims);
    const tensor_strides k_pad_strides = make_contiguous_strides(k_pad_dims);
    const tensor_strides v_pad_strides = make_contiguous_strides(v_pad_dims);
    ggml_cuda_pool & pool = ctx.pool();
    std::unique_ptr<ggml_cuda_pool_alloc<T>> q_padded;
    std::unique_ptr<ggml_cuda_pool_alloc<T>> k_padded;
    std::unique_ptr<ggml_cuda_pool_alloc<T>> v_padded;
    std::unique_ptr<ggml_cuda_pool_alloc<T>> q_bhsd;
    std::unique_ptr<ggml_cuda_pool_alloc<T>> k_bhsd;
    std::unique_ptr<ggml_cuda_pool_alloc<T>> v_bhsd;
    const int batch = q_dims.batch;
    const int num_q_heads = q_dims.heads;
    const int num_k_heads = k_dims.heads;
    GGML_ASSERT(num_q_heads % num_k_heads == 0);
    const int num_kv_groups = num_q_heads / num_k_heads;
    const bool require_dim_pad = head_dim_padded != head_dim;
    const bool need_q_pad = require_dim_pad || dump_prefix != nullptr || debug_quant_host;
    const bool need_k_pad = require_dim_pad || dump_prefix != nullptr || debug_quant_host;
    const tensor_strides q_int8_strides = {
        head_dim_padded,
        (int64_t) head_dim_padded * qo_len,
        (int64_t) head_dim_padded * qo_len * num_q_heads,
    };
    const tensor_strides k_int8_strides = {
        head_dim_padded,
        (int64_t) head_dim_padded * kv_len,
        (int64_t) head_dim_padded * kv_len * num_k_heads,
    };
    const tensor_strides v_transposed_strides = {
        head_dim_padded,
        (int64_t) head_dim_padded * kv_len_padded,
        (int64_t) head_dim_padded * kv_len_padded * num_k_heads,
    };
    const tensor_strides v_bhsd_input_strides = {
        head_dim_padded,
        (int64_t) head_dim_padded * kv_len,
        (int64_t) head_dim_padded * kv_len * num_k_heads,
    };
    const tensor_strides q_input_strides = need_q_pad ? q_pad_strides : q_strides;
    const tensor_strides k_input_strides = need_k_pad ? k_pad_strides : k_strides;
    const tensor_dims & q_src_dims = need_q_pad ? q_pad_dims : q_dims;
    const tensor_dims & k_src_dims = need_k_pad ? k_pad_dims : k_dims;
    const T * q_input_ptr = static_cast<const T *>(q->data);
    T * q_padded_ptr = nullptr;
    if (need_q_pad) {
        q_padded = std::make_unique<ggml_cuda_pool_alloc<T>>(pool, q_pad_dims.head_dim * q_pad_dims.seq_len * q_pad_dims.heads * q_pad_dims.batch);
        q_padded_ptr = q_padded->get();
        copy_pad_tensor(
            static_cast<const T *>(q->data),
            q_dims, q_strides,
            q_pad_dims, q_pad_strides,
            q_padded_ptr,
            stream);
        q_input_ptr = q_padded_ptr;
        if (dump_prefix) {
            sage_debug_dump(dump_prefix, "q_f16", q_padded_ptr,
                (size_t) q_pad_dims.head_dim * q_pad_dims.seq_len * q_pad_dims.heads * q_pad_dims.batch * sizeof(T),
                stream);
        }
    } else if (dump_prefix) {
        sage_debug_dump(dump_prefix, "q_f16", q->data,
            (size_t) q_dims.head_dim * q_dims.seq_len * q_dims.heads * q_dims.batch * sizeof(T),
            stream);
    }
    const T * k_input_ptr = static_cast<const T *>(k->data);
    T * k_padded_ptr = nullptr;
    if (need_k_pad) {
        k_padded = std::make_unique<ggml_cuda_pool_alloc<T>>(pool, k_pad_dims.head_dim * k_pad_dims.seq_len * k_pad_dims.heads * k_pad_dims.batch);
        k_padded_ptr = k_padded->get();
        copy_pad_tensor(
            static_cast<const T *>(k->data),
            k_dims, k_strides,
            k_pad_dims, k_pad_strides,
            k_padded_ptr,
            stream);
        k_input_ptr = k_padded_ptr;
        if (dump_prefix) {
            sage_debug_dump(dump_prefix, "k_f16", k_padded_ptr,
                (size_t) k_pad_dims.head_dim * k_pad_dims.seq_len * k_pad_dims.heads * k_pad_dims.batch * sizeof(T),
                stream);
        }
    } else if (dump_prefix) {
        sage_debug_dump(dump_prefix, "k_f16", k->data,
            (size_t) k_dims.head_dim * k_dims.seq_len * k_dims.heads * k_dims.batch * sizeof(T),
            stream);
    }

    q_bhsd = std::make_unique<ggml_cuda_pool_alloc<T>>(pool, q_pad_dims.head_dim * q_pad_dims.seq_len * q_pad_dims.heads * q_pad_dims.batch);
    CUDA_CHECK(cudaMemsetAsync(q_bhsd->get(), 0, (size_t) q_pad_dims.head_dim * q_pad_dims.seq_len * q_pad_dims.heads * q_pad_dims.batch * sizeof(T), stream));
    if (force_host_permute) {
        convert_dshb_to_bhsd_host(
            q_input_ptr,
            q_bhsd->get(),
            q_src_dims.head_dim,
            q_src_dims.seq_len,
            q_src_dims.heads,
            q_src_dims.batch,
            stream);
    } else {
        convert_dshb_to_bhsd_device(
            q_input_ptr,
            q_bhsd->get(),
            q_src_dims.head_dim,
            q_src_dims.seq_len,
            q_src_dims.heads,
            q_src_dims.batch,
            stream);
    }
    q_input_ptr = q_bhsd->get();

    k_bhsd = std::make_unique<ggml_cuda_pool_alloc<T>>(pool, k_pad_dims.head_dim * k_pad_dims.seq_len * k_pad_dims.heads * k_pad_dims.batch);
    CUDA_CHECK(cudaMemsetAsync(k_bhsd->get(), 0, (size_t) k_pad_dims.head_dim * k_pad_dims.seq_len * k_pad_dims.heads * k_pad_dims.batch * sizeof(T), stream));
    if (force_host_permute) {
        convert_dshb_to_bhsd_host(
            k_input_ptr,
            k_bhsd->get(),
            k_src_dims.head_dim,
            k_src_dims.seq_len,
            k_src_dims.heads,
            k_src_dims.batch,
            stream);
    } else {
        convert_dshb_to_bhsd_device(
            k_input_ptr,
            k_bhsd->get(),
            k_src_dims.head_dim,
            k_src_dims.seq_len,
            k_src_dims.heads,
            k_src_dims.batch,
            stream);
    }
    k_input_ptr = k_bhsd->get();

    const bool need_v_pad = require_dim_pad || dump_prefix != nullptr || force_host_permute;
    const tensor_dims & v_src_dims = need_v_pad ? v_pad_dims : v_dims;
    T * v_padded_ptr = nullptr;
    if (need_v_pad) {
        v_padded = std::make_unique<ggml_cuda_pool_alloc<T>>(pool, v_pad_dims.head_dim * v_pad_dims.seq_len * v_pad_dims.heads * v_pad_dims.batch);
        v_padded_ptr = v_padded->get();
        copy_pad_tensor(
            static_cast<const T *>(v->data),
            v_dims, v_strides,
            v_pad_dims, v_pad_strides,
            v_padded_ptr,
            stream);
    }
    if (dump_prefix && v_padded_ptr) {
        sage_debug_dump(dump_prefix, "v_f16", v_padded_ptr,
            (size_t) v_pad_dims.head_dim * v_pad_dims.seq_len * v_pad_dims.heads * v_pad_dims.batch * sizeof(T),
            stream);
    }

    v_bhsd = std::make_unique<ggml_cuda_pool_alloc<T>>(pool, v_pad_dims.head_dim * v_pad_dims.seq_len * v_pad_dims.heads * v_pad_dims.batch);
    CUDA_CHECK(cudaMemsetAsync(v_bhsd->get(), 0, (size_t) v_pad_dims.head_dim * v_pad_dims.seq_len * v_pad_dims.heads * v_pad_dims.batch * sizeof(T), stream));
    if (force_host_permute) {
        convert_dshb_to_bhsd_host(
            v_padded_ptr ? v_padded_ptr : static_cast<const T *>(v->data),
            v_bhsd->get(),
            v_src_dims.head_dim,
            v_src_dims.seq_len,
            v_src_dims.heads,
            v_src_dims.batch,
            stream);
    } else {
        convert_dshb_to_bhsd_device(
            v_padded_ptr ? v_padded_ptr : static_cast<const T *>(v->data),
            v_bhsd->get(),
            v_src_dims.head_dim,
            v_src_dims.seq_len,
            v_src_dims.heads,
            v_src_dims.batch,
            stream);
    }
    const T * v_bhsd_ptr = v_bhsd->get();
    sage_kernel_debug_q q_kernel_debug = {};
    size_t q_kernel_debug_slots = 0;
    std::unique_ptr<ggml_cuda_pool_alloc<uint32_t>> dbg_q_token_storage;
    std::unique_ptr<ggml_cuda_pool_alloc<uint32_t>> dbg_q_scale_idx_storage;
    std::unique_ptr<ggml_cuda_pool_alloc<float>> dbg_q_scale_val_storage;
    std::unique_ptr<ggml_cuda_pool_alloc<float>> dbg_q_iter_k_scale_val_storage;
    std::unique_ptr<ggml_cuda_pool_alloc<float>> dbg_q_iter_dequant_storage;
    std::unique_ptr<ggml_cuda_pool_alloc<float>> dbg_q_iter_sm_scale_storage;
    std::unique_ptr<ggml_cuda_pool_alloc<uint32_t>> dbg_q_k_scale_idx_storage;
    std::unique_ptr<ggml_cuda_pool_alloc<float>> dbg_q_k_scale_val_storage;
    std::unique_ptr<ggml_cuda_pool_alloc<float>> dbg_q_dequant_storage;
    std::unique_ptr<ggml_cuda_pool_alloc<float>> dbg_q_sm_scale_storage;
    if (debug_kernel) {
        const int warps_per_block = CTA_Q / WARP_Q;
        q_kernel_debug.stride = div_ceil64(qo_len, CTA_Q) * warps_per_block;
        q_kernel_debug_slots = (size_t) batch * num_q_heads * q_kernel_debug.stride;
        if (q_kernel_debug_slots > 0) {
            dbg_q_token_storage = std::make_unique<ggml_cuda_pool_alloc<uint32_t>>(pool, q_kernel_debug_slots);
            dbg_q_scale_idx_storage = std::make_unique<ggml_cuda_pool_alloc<uint32_t>>(pool, q_kernel_debug_slots);
            dbg_q_scale_val_storage = std::make_unique<ggml_cuda_pool_alloc<float>>(pool, q_kernel_debug_slots);
            q_kernel_debug.token_start = dbg_q_token_storage->get();
            q_kernel_debug.scale_index = dbg_q_scale_idx_storage->get();
            q_kernel_debug.scale_value = dbg_q_scale_val_storage->get();
            if (debug_kernel_iters) {
                const int max_k_iters = std::max<int>(1, (int) div_ceil64(kv_len, CTA_K));
                q_kernel_debug.iter_stride = q_kernel_debug.stride;
                q_kernel_debug.iter_count = max_k_iters;
                const size_t iter_slots = q_kernel_debug_slots * (size_t) max_k_iters;
                if (iter_slots > 0) {
                    dbg_q_iter_k_scale_val_storage = std::make_unique<ggml_cuda_pool_alloc<float>>(pool, iter_slots);
                    dbg_q_iter_dequant_storage = std::make_unique<ggml_cuda_pool_alloc<float>>(pool, iter_slots);
                    dbg_q_iter_sm_scale_storage = std::make_unique<ggml_cuda_pool_alloc<float>>(pool, iter_slots);
                    q_kernel_debug.k_scale_iter = dbg_q_iter_k_scale_val_storage->get();
                    q_kernel_debug.dequant_iter = dbg_q_iter_dequant_storage->get();
                    q_kernel_debug.sm_scale_iter = dbg_q_iter_sm_scale_storage->get();
                }
            }
            dbg_q_k_scale_idx_storage = std::make_unique<ggml_cuda_pool_alloc<uint32_t>>(pool, q_kernel_debug_slots);
            dbg_q_k_scale_val_storage = std::make_unique<ggml_cuda_pool_alloc<float>>(pool, q_kernel_debug_slots);
            dbg_q_dequant_storage = std::make_unique<ggml_cuda_pool_alloc<float>>(pool, q_kernel_debug_slots);
            dbg_q_sm_scale_storage = std::make_unique<ggml_cuda_pool_alloc<float>>(pool, q_kernel_debug_slots);
            q_kernel_debug.k_scale_index = dbg_q_k_scale_idx_storage->get();
            q_kernel_debug.k_scale_value = dbg_q_k_scale_val_storage->get();
            q_kernel_debug.dequant_scale = dbg_q_dequant_storage->get();
            q_kernel_debug.sm_scale_value = dbg_q_sm_scale_storage->get();
        }
    }

    const int q_blocks = div_ceil64(qo_len, CTA_Q) * (CTA_Q / WARPQ);
    const int k_blocks = div_ceil64(kv_len, CTA_K) * (CTA_K / WARP_K);
    const int q_scale_cols = per_thread_qk ? q_blocks * 8 : ((qo_len + BLKQ - 1) / BLKQ) * (BLKQ / WARPQ);
    const int k_scale_cols = per_thread_qk ? k_blocks * 4 : (kv_len + BLKK - 1) / BLKK;

    ggml_cuda_pool_alloc<int8_t> q_int8(pool, head_dim_padded * qo_len * num_q_heads * batch);
    ggml_cuda_pool_alloc<int8_t> k_int8(pool, k_pad_dims.head_dim * kv_len * num_k_heads * batch);
    ggml_cuda_pool_alloc<float>  q_scale(pool, (size_t) batch * num_q_heads * q_scale_cols);
    ggml_cuda_pool_alloc<float>  k_scale(pool, (size_t) batch * num_k_heads * k_scale_cols);

    const size_t q_scale_elems = (size_t) batch * num_q_heads * q_scale_cols;
    ggml_cuda_sage::quant_debug_config q_quant_debug = {};
    size_t q_quant_debug_elems = 0;
    std::unique_ptr<ggml_cuda_pool_alloc<uint32_t>> dbg_q_quant_token_storage;
    std::unique_ptr<ggml_cuda_pool_alloc<float>> dbg_q_quant_amax_storage;
    std::unique_ptr<ggml_cuda_pool_alloc<float>> dbg_q_quant_scale_storage;
    std::unique_ptr<ggml_cuda_pool_alloc<half>> dbg_q_quant_samples_storage;
    if (debug_quant_dev && !per_thread_qk) {
        q_quant_debug.stride = q_scale_cols;
        q_quant_debug_elems = q_scale_elems;
        dbg_q_quant_token_storage = std::make_unique<ggml_cuda_pool_alloc<uint32_t>>(pool, q_quant_debug_elems);
        dbg_q_quant_amax_storage = std::make_unique<ggml_cuda_pool_alloc<float>>(pool, q_quant_debug_elems);
        dbg_q_quant_scale_storage = std::make_unique<ggml_cuda_pool_alloc<float>>(pool, q_quant_debug_elems);
        q_quant_debug.token_start = dbg_q_quant_token_storage->get();
        q_quant_debug.amax = dbg_q_quant_amax_storage->get();
        q_quant_debug.scale = dbg_q_quant_scale_storage->get();
        const uint32_t sample_tokens = BLKQ; // capture full block
        const size_t sample_stride = (size_t) sample_tokens * head_dim_padded;
        dbg_q_quant_samples_storage = std::make_unique<ggml_cuda_pool_alloc<half>>(pool, q_quant_debug_elems * sample_stride);
        q_quant_debug.samples = dbg_q_quant_samples_storage->get();
        q_quant_debug.sample_stride = sample_stride;
        fprintf(stderr,
                "SAGE_Q_DEV_DEBUG alloc: tokens=%zu stride=%u sample_ptr=%p token_ptr=%p\n",
                q_quant_debug_elems,
                q_quant_debug.sample_stride,
                (void *) q_quant_debug.samples,
                (void *) q_quant_debug.token_start);
    }

    const T * q_quant_src = q_input_ptr;
    ggml_cuda_sage::quant_debug_config k_quant_debug = {};
    size_t k_quant_debug_elems = 0;
    std::unique_ptr<ggml_cuda_pool_alloc<uint32_t>> dbg_k_quant_token_storage;
    std::unique_ptr<ggml_cuda_pool_alloc<float>> dbg_k_quant_amax_storage;
    std::unique_ptr<ggml_cuda_pool_alloc<float>> dbg_k_quant_scale_storage;
    std::unique_ptr<ggml_cuda_pool_alloc<half>> dbg_k_quant_samples_storage;
    if (sage_force_host_quant()) {
        quantize_q_per_warp_host(
            q_quant_src,
            q_int8.get(),
            q_scale.get(),
            batch,
            num_q_heads,
            qo_len,
            head_dim_padded,
            stream);
    } else if (per_thread_qk) {
        switch (head_dim_padded) {
            case 64:
                quantize_q_per_thread<T, 64>(
                    q_quant_src,
                    q_int8.get(),
                    q_scale.get(),
                    batch,
                    num_q_heads,
                    qo_len,
                    stream);
                break;
            case 128:
                quantize_q_per_thread<T, 128>(
                    q_quant_src,
                    q_int8.get(),
                    q_scale.get(),
                    batch,
                    num_q_heads,
                    qo_len,
                    stream);
                break;
            default:
                GGML_ABORT("unsupported head_dim_padded");
        }
    } else {
        switch (head_dim_padded) {
            case 64:
                quantize_q_per_warp<T, 64>(
                    q_quant_src,
                    q_int8.get(),
                    q_scale.get(),
                    batch,
                    num_q_heads,
                    qo_len,
                    stream,
                    q_quant_debug);
                break;
            case 128:
                quantize_q_per_warp<T, 128>(
                    q_quant_src,
                    q_int8.get(),
                    q_scale.get(),
                    batch,
                    num_q_heads,
                    qo_len,
                    stream,
                    q_quant_debug);
                break;
            default:
            GGML_ABORT("unsupported head_dim_padded");
        }
    }
    if (debug_quant_dev && !per_thread_qk) {
        k_quant_debug.stride = k_scale_cols;
        k_quant_debug_elems = (size_t) batch * num_k_heads * k_scale_cols;
        if (k_quant_debug_elems > 0) {
            dbg_k_quant_token_storage = std::make_unique<ggml_cuda_pool_alloc<uint32_t>>(pool, k_quant_debug_elems);
            dbg_k_quant_amax_storage  = std::make_unique<ggml_cuda_pool_alloc<float>>(pool, k_quant_debug_elems);
            dbg_k_quant_scale_storage = std::make_unique<ggml_cuda_pool_alloc<float>>(pool, k_quant_debug_elems);
            k_quant_debug.token_start = dbg_k_quant_token_storage->get();
            k_quant_debug.amax = dbg_k_quant_amax_storage->get();
            k_quant_debug.scale = dbg_k_quant_scale_storage->get();
            const uint32_t sample_tokens = BLKK;
            const size_t sample_stride = (size_t) sample_tokens * head_dim_padded;
            dbg_k_quant_samples_storage = std::make_unique<ggml_cuda_pool_alloc<half>>(pool, k_quant_debug_elems * sample_stride);
            k_quant_debug.samples = dbg_k_quant_samples_storage->get();
            k_quant_debug.sample_stride = sample_stride;
        }
    }
    if (debug_quant) {
        ggml_cuda_pool_alloc<float> q_amax_dev(pool, q_scale_elems);
        const int blocks_q = ((qo_len + BLKQ - 1) / BLKQ) * (BLKQ / WARPQ);
        dim3 grid_dbg(blocks_q, num_q_heads, batch);
        dim3 block_dbg(128);
        switch (head_dim_padded) {
            case 64:
                compute_q_block_amax_kernel<T, 64><<<grid_dbg, block_dbg, 0, stream>>>(
                    q_quant_src, q_amax_dev.get(), qo_len,
                    to_u32(q_input_strides.stride_batch), to_u32(q_input_strides.stride_seq), to_u32(q_input_strides.stride_head), q_scale_cols);
                break;
            case 128:
                compute_q_block_amax_kernel<T, 128><<<grid_dbg, block_dbg, 0, stream>>>(
                    q_quant_src, q_amax_dev.get(), qo_len,
                    to_u32(q_input_strides.stride_batch), to_u32(q_input_strides.stride_seq), to_u32(q_input_strides.stride_head), q_scale_cols);
                break;
        }
        CUDA_CHECK(cudaGetLastError());
        std::vector<float> host_amax(q_scale_elems);
        std::vector<float> host_scale(q_scale_elems);
        CUDA_CHECK(cudaMemcpyAsync(host_amax.data(), q_amax_dev.get(), q_scale_elems * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(host_scale.data(), q_scale.get(), q_scale_elems * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        float max_diff = 0.0f;
        size_t max_idx = 0;
        for (size_t i = 0; i < q_scale_elems; ++i) {
            const float actual_scale = host_amax[i] / 127.0f;
            const float diff = fabsf(actual_scale - host_scale[i]);
            if (diff > max_diff) {
                max_diff = diff;
                max_idx = i;
            }
        }
        const int cols = q_scale_cols;
        const int heads_total = num_q_heads;
        const int batch_idx = max_idx / (heads_total * cols);
        const int head_idx = (max_idx / cols) % heads_total;
        const int block_idx = max_idx % cols;
        const int block128 = block_idx / (BLKQ / WARPQ);
        const int offset = block_idx % (BLKQ / WARPQ);
        const int token_start = block128 * BLKQ + offset * WARPQ;
        fprintf(stderr,
                "SAGE_Q_AMAX max_diff=%g batch=%d head=%d block=%d token_start=%d actual=%g scale=%g\n",
                max_diff,
                batch_idx,
                head_idx,
                block_idx,
                token_start,
                host_amax[max_idx],
                host_scale[max_idx]);
        if (dump_prefix) {
            sage_debug_dump(dump_prefix, "q_amax", q_amax_dev.get(), q_scale_elems * sizeof(float), stream);
        }
    }
    if (((debug_quant && sage_force_simple_quant()) || debug_quant_host) && q_padded_ptr) {
        fprintf(stderr, "SAGE_Q_SIMPLE host check start\n");
        debug_compare_q_scale_host(
            q_padded_ptr,
            q_pad_dims,
            q_scale.get(),
            q_scale_cols,
            stream);
    }
    if (debug_quant_dev && !per_thread_qk && q_quant_debug.token_start != nullptr) {
        std::vector<uint32_t> host_tokens(q_quant_debug_elems);
        std::vector<float> host_amax_dbg(q_quant_debug_elems);
        std::vector<float> host_scale_dbg(q_quant_debug_elems);
        std::vector<ggml_fp16_t> host_samples;
        if (q_quant_debug.samples && q_quant_debug.sample_stride > 0) {
            host_samples.resize(q_quant_debug_elems * q_quant_debug.sample_stride);
        }
        CUDA_CHECK(cudaMemcpyAsync(host_tokens.data(), q_quant_debug.token_start, q_quant_debug_elems * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(host_amax_dbg.data(), q_quant_debug.amax, q_quant_debug_elems * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(host_scale_dbg.data(), q_quant_debug.scale, q_quant_debug_elems * sizeof(float), cudaMemcpyDeviceToHost, stream));
        if (q_quant_debug.samples && q_quant_debug.sample_stride > 0) {
            CUDA_CHECK(cudaMemcpyAsync(host_samples.data(), q_quant_debug.samples, host_samples.size() * sizeof(ggml_fp16_t), cudaMemcpyDeviceToHost, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const int warps_per_block = BLKQ / WARPQ;
        bool mismatch = false;
        int printed = 0;
        const char * sample_dump_prefix = getenv("GGML_SAGE_SAMPLE_PREFIX");
        const int sample_head_filter = getenv("GGML_SAGE_SAMPLE_HEAD") ? atoi(getenv("GGML_SAGE_SAMPLE_HEAD")) : -1;
        const int sample_blk_filter = getenv("GGML_SAGE_SAMPLE_BLOCK") ? atoi(getenv("GGML_SAGE_SAMPLE_BLOCK")) : -1;
        const int sample_warp_filter = getenv("GGML_SAGE_SAMPLE_WARP") ? atoi(getenv("GGML_SAGE_SAMPLE_WARP")) : -1;
        const int sample_limit = getenv("GGML_SAGE_SAMPLE_LIMIT") ? atoi(getenv("GGML_SAGE_SAMPLE_LIMIT")) : 4;
        fprintf(stderr, "SAGE_Q_DEV_DEBUG sample env=%s ptr=%p stride=%u host_samples=%zu filter(head=%d blk=%d warp=%d) limit=%d\n",
                sample_dump_prefix ? sample_dump_prefix : "(null)",
                (void *) q_quant_debug.samples,
                q_quant_debug.sample_stride,
                host_samples.size(),
                sample_head_filter,
                sample_blk_filter,
                sample_warp_filter,
                sample_limit);
        int sample_dumps = 0;
        for (int b = 0; b < batch && !mismatch; ++b) {
            for (int h = 0; h < num_q_heads && !mismatch; ++h) {
                for (int blk = 0; blk < div_ceil64(qo_len, BLKQ) && !mismatch; ++blk) {
                    for (int warp = 0; warp < warps_per_block && !mismatch; ++warp) {
                        const size_t idx =
                            (size_t) b * num_q_heads * q_scale_cols +
                            (size_t) h * q_scale_cols +
                            (size_t) blk * warps_per_block + warp;
                        if (idx >= q_quant_debug_elems) {
                            continue;
                        }
                        const int token_expected = blk * BLKQ + warp * WARPQ;
                        if (token_expected >= qo_len) {
                            continue;
                        }
                        const uint32_t token_recorded = host_tokens[idx];
                        float recomputed_scale = host_scale_dbg[idx];
                        if (!host_samples.empty() && q_quant_debug.sample_stride > 0) {
                            const size_t sample_offset = idx * q_quant_debug.sample_stride;
                            const int max_tokens = q_quant_debug.sample_stride / head_dim_padded;
                            const int token_end = std::min(token_recorded + (uint32_t) max_tokens, (uint32_t) qo_len);
                            float local_max = 1e-7f;
                            for (uint32_t tok = token_recorded; tok < (uint32_t) token_end; ++tok) {
                                const size_t tok_idx = sample_offset + (size_t) (tok - token_recorded) * head_dim_padded;
                                for (int d = 0; d < head_dim_padded; ++d) {
                                    local_max = std::max(local_max, fabsf(ggml_fp16_to_fp32(host_samples[tok_idx + d])));
                                }
                            }
                            recomputed_scale = local_max / 127.0f;
                        }
                        if (printed < 4) {
                            fprintf(stderr,
                                    "SAGE_Q_DEV_DEBUG batch=%d head=%d blk=%d warp=%d token=%u expected=%d scale=%g amax=%g recomputed=%g\n",
                                    b, h, blk, warp,
                                    token_recorded,
                                    token_expected,
                                    host_scale_dbg[idx],
                                    host_amax_dbg[idx],
                                    recomputed_scale);
                            printed++;
                        }
                        if (sample_dump_prefix && !host_samples.empty() && sample_dumps < sample_limit &&
                                (sample_head_filter < 0 || h == sample_head_filter) &&
                                (sample_blk_filter < 0 || blk == sample_blk_filter) &&
                                (sample_warp_filter < 0 || warp == sample_warp_filter)) {
                            const size_t sample_offset = idx * q_quant_debug.sample_stride;
                            const size_t sample_bytes = (size_t) q_quant_debug.sample_stride * sizeof(ggml_fp16_t);
                            char path_bin[512];
                            snprintf(path_bin, sizeof(path_bin), "%s.b%d_h%d_blk%d_w%d.bin",
                                     sample_dump_prefix, b, h, blk, warp);
                            FILE * fp = fopen(path_bin, "wb");
                            if (fp) {
                                fwrite(host_samples.data() + sample_offset, 1, sample_bytes, fp);
                                fclose(fp);
                                char path_meta[512];
                                snprintf(path_meta, sizeof(path_meta), "%s.b%d_h%d_blk%d_w%d.meta",
                                         sample_dump_prefix, b, h, blk, warp);
                                FILE * meta = fopen(path_meta, "w");
                                if (meta) {
                                    fprintf(meta, "batch %d\nhead %d\nblock %d\nwarp %d\n"
                                                  "token_start %u\nhead_dim %d\nsample_stride %u\n",
                                            b, h, blk, warp, token_recorded, head_dim_padded, q_quant_debug.sample_stride);
                                    fclose(meta);
                                }
                                fprintf(stderr, "SAGE_Q_DEV_DEBUG dumped samples to %s\n", path_bin);
                            }
                            sample_dumps++;
                        }
                        if (fabsf(recomputed_scale - host_scale_dbg[idx]) > 1e-4f) {
                            fprintf(stderr,
                                    "SAGE_Q_DEV_DEBUG scale mismatch batch=%d head=%d blk=%d warp=%d recorded=%g recomputed=%g\n",
                                    b, h, blk, warp,
                                    host_scale_dbg[idx],
                                    recomputed_scale);
                            mismatch = true;
                        }
                        if (!mismatch && (int) token_recorded != token_expected) {
                            fprintf(stderr,
                                    "SAGE_Q_DEV_DEBUG mismatch batch=%d head=%d blk=%d warp=%d recorded_token=%u expected_token=%d\n",
                                    b, h, blk, warp,
                                    token_recorded,
                                    token_expected);
                            mismatch = true;
                        }
                    }
                }
            }
        }
        if (!mismatch) {
            fprintf(stderr, "SAGE_Q_DEV_DEBUG token order verified for %zu entries\n", q_quant_debug_elems);
        }
    }
    if (dump_prefix) {
        sage_debug_dump(dump_prefix, "q_scale", q_scale.get(), q_scale_elems * sizeof(float), stream);
        if (dump_int8_qk) {
            sage_debug_dump(dump_prefix, "q_int8", q_int8.get(),
                (size_t) q_pad_dims.head_dim * qo_len * num_q_heads * batch * sizeof(int8_t),
                stream);
        }
    }

    const T * k_quant_src = k_input_ptr;
    std::unique_ptr<ggml_cuda_pool_alloc<T>> k_mean_storage;
    T * k_mean_ptr = nullptr;
    if (smooth_k) {
        k_mean_storage = std::make_unique<ggml_cuda_pool_alloc<T>>(pool, (size_t) head_dim_padded * num_k_heads * batch);
        tensor_strides mean_strides = {
            0,
            head_dim_padded,
            head_dim_padded * num_k_heads,
        };
        const tensor_dims k_mean_dims = need_k_pad ? k_pad_dims : k_dims;
        compute_k_mean(k_input_ptr, k_mean_dims, k_input_strides, mean_strides, k_mean_storage->get(), stream);
        k_mean_ptr = k_mean_storage->get();
        if (dump_prefix) {
            sage_debug_dump(dump_prefix, "k_mean", k_mean_ptr,
                (size_t) head_dim_padded * num_k_heads * batch * sizeof(T),
                stream);
        }
    }

    if (sage_force_host_quant()) {
        quantize_k_per_block_host(
            k_quant_src,
            smooth_k ? k_mean_ptr : nullptr,
            k_int8.get(),
            k_scale.get(),
            batch,
            num_k_heads,
            kv_len,
            head_dim_padded,
            stream);
    } else if (per_thread_qk) {
        switch (head_dim_padded) {
            case 64:
                if (smooth_k) {
                    quantize_k_per_thread<T, 64, true>(
                        k_quant_src, k_mean_ptr,
                        k_int8.get(), k_scale.get(),
                        batch, num_k_heads, kv_len, stream);
                } else {
                    quantize_k_per_thread<T, 64, false>(
                        k_quant_src, nullptr,
                        k_int8.get(), k_scale.get(),
                        batch, num_k_heads, kv_len, stream);
                }
                break;
            case 128:
                if (smooth_k) {
                    quantize_k_per_thread<T, 128, true>(
                        k_quant_src, k_mean_ptr,
                        k_int8.get(), k_scale.get(),
                        batch, num_k_heads, kv_len, stream);
                } else {
                    quantize_k_per_thread<T, 128, false>(
                        k_quant_src, nullptr,
                        k_int8.get(), k_scale.get(),
                        batch, num_k_heads, kv_len, stream);
                }
                break;
            default:
                GGML_ABORT("unsupported head_dim_padded");
        }
    } else {
        switch (head_dim_padded) {
            case 64:
                if (smooth_k) {
                    quantize_k_per_block<T, 64, true>(
                        k_quant_src, k_mean_ptr,
                        k_int8.get(), k_scale.get(),
                        batch, num_k_heads, kv_len, stream, k_quant_debug);
                } else {
                    quantize_k_per_block<T, 64, false>(
                        k_quant_src, nullptr,
                        k_int8.get(), k_scale.get(),
                        batch, num_k_heads, kv_len, stream, k_quant_debug);
                }
                break;
            case 128:
                if (smooth_k) {
                    quantize_k_per_block<T, 128, true>(
                        k_quant_src, k_mean_ptr,
                        k_int8.get(), k_scale.get(),
                        batch, num_k_heads, kv_len, stream, k_quant_debug);
                } else {
                    quantize_k_per_block<T, 128, false>(
                        k_quant_src, nullptr,
                        k_int8.get(), k_scale.get(),
                        batch, num_k_heads, kv_len, stream, k_quant_debug);
                }
                break;
            default:
                GGML_ABORT("unsupported head_dim_padded");
        }
    }
    const size_t k_scale_elems = (size_t) batch * num_k_heads * k_scale_cols;
    if (((debug_quant && sage_force_simple_quant()) || debug_quant_host) && k_padded_ptr) {
        fprintf(stderr, "SAGE_K_SIMPLE host check start\n");
        debug_compare_k_scale_host(
            k_padded_ptr,
            k_pad_dims,
            k_scale.get(),
            k_scale_cols,
            smooth_k,
            k_mean_ptr,
            stream);
    }
    if (debug_quant_dev && !per_thread_qk && k_quant_debug.token_start != nullptr) {
        std::vector<uint32_t> host_tokens(k_quant_debug_elems);
        std::vector<float> host_amax_dbg(k_quant_debug_elems);
        std::vector<float> host_scale_dbg(k_quant_debug_elems);
        std::vector<ggml_fp16_t> host_k_samples;
        if (k_quant_debug.samples && k_quant_debug.sample_stride > 0) {
            host_k_samples.resize(k_quant_debug_elems * k_quant_debug.sample_stride);
        }
        CUDA_CHECK(cudaMemcpyAsync(host_tokens.data(), k_quant_debug.token_start, k_quant_debug_elems * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(host_amax_dbg.data(), k_quant_debug.amax, k_quant_debug_elems * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(host_scale_dbg.data(), k_quant_debug.scale, k_quant_debug_elems * sizeof(float), cudaMemcpyDeviceToHost, stream));
        if (k_quant_debug.samples && k_quant_debug.sample_stride > 0) {
            CUDA_CHECK(cudaMemcpyAsync(host_k_samples.data(), k_quant_debug.samples, host_k_samples.size() * sizeof(ggml_fp16_t), cudaMemcpyDeviceToHost, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        bool mismatch = false;
        size_t printed = 0;
        const int blocks_k = div_ceil64(kv_len, BLKK);
        const char * sample_prefix = getenv("GGML_SAGE_SAMPLE_PREFIX");
        const int sample_head_filter = getenv("GGML_SAGE_SAMPLE_HEAD") ? atoi(getenv("GGML_SAGE_SAMPLE_HEAD")) : -1;
        const int sample_blk_filter = getenv("GGML_SAGE_SAMPLE_BLOCK") ? atoi(getenv("GGML_SAGE_SAMPLE_BLOCK")) : -1;
        const int sample_limit = getenv("GGML_SAGE_SAMPLE_LIMIT") ? atoi(getenv("GGML_SAGE_SAMPLE_LIMIT")) : 4;
        int sample_dumps = 0;
        for (int b = 0; b < batch && !mismatch; ++b) {
            for (int h = 0; h < num_k_heads && !mismatch; ++h) {
                for (int blk = 0; blk < blocks_k && !mismatch; ++blk) {
                    const size_t idx =
                        (size_t) b * num_k_heads * k_scale_cols +
                        (size_t) h * k_scale_cols +
                        blk;
                    if (idx >= k_quant_debug_elems) {
                        continue;
                    }
                    const int token_expected = blk * BLKK;
                    if (token_expected >= kv_len) {
                        continue;
                    }
                    const uint32_t token_recorded = host_tokens[idx];
                    if (printed < 4) {
                        fprintf(stderr,
                                "SAGE_K_DEV_DEBUG batch=%d head=%d blk=%d token=%u expected=%d scale=%g amax=%g\n",
                                b, h, blk,
                                token_recorded,
                                token_expected,
                                host_scale_dbg[idx],
                                host_amax_dbg[idx]);
                        printed++;
                    }
                    if (token_recorded != (uint32_t) token_expected) {
                        fprintf(stderr,
                                "SAGE_K_DEV_DEBUG mismatch batch=%d head=%d blk=%d recorded_token=%u expected_token=%d\n",
                                b, h, blk,
                                token_recorded,
                                token_expected);
                        mismatch = true;
                        break;
                    }
                    if (sample_prefix && !host_k_samples.empty() &&
                            (sample_head_filter == -1 || sample_head_filter == h) &&
                            (sample_blk_filter == -1 || sample_blk_filter == blk) &&
                            sample_dumps < sample_limit) {
                        const size_t sample_offset = idx * k_quant_debug.sample_stride;
                        const int token_count = std::min<int>(BLKK, kv_len - token_expected);
                        const std::string path = std::string(sample_prefix) +
                            ".k.b" + std::to_string(b) +
                            "_h" + std::to_string(h) +
                            "_blk" + std::to_string(blk) + ".bin";
                        FILE * fp = fopen(path.c_str(), "wb");
                        if (fp) {
                            fwrite(host_k_samples.data() + sample_offset, sizeof(ggml_fp16_t),
                                   (size_t) token_count * head_dim_padded, fp);
                            fclose(fp);
                            fprintf(stderr, "SAGE_K_DEV_DEBUG wrote sample %s tokens=%d\n", path.c_str(), token_count);
                        }
                        sample_dumps++;
                    }
                }
            }
        }
        if (!mismatch) {
            fprintf(stderr, "SAGE_K_DEV_DEBUG token order verified for %zu entries\n", k_quant_debug_elems);
        }
    }
    if (dump_prefix) {
        sage_debug_dump(dump_prefix, "k_scale", k_scale.get(), k_scale_elems * sizeof(float), stream);
        if (dump_int8_qk) {
            sage_debug_dump(dump_prefix, "k_int8", k_int8.get(),
                (size_t) k_pad_dims.head_dim * kv_len * num_k_heads * batch * sizeof(int8_t),
                stream);
        }
    }
    if (debug_quant) {
        ggml_cuda_pool_alloc<float> k_amax_dev(pool, k_scale_elems);
        const int blocks_k = (kv_len + BLKK - 1) / BLKK;
        dim3 grid_dbg_k(blocks_k, num_k_heads, batch);
        dim3 block_dbg_k(128);
        switch (head_dim_padded) {
            case 64:
                compute_k_block_amax_kernel<T, 64><<<grid_dbg_k, block_dbg_k, 0, stream>>>(
                    k_quant_src, k_amax_dev.get(), kv_len,
                    to_u32(k_input_strides.stride_batch), to_u32(k_input_strides.stride_seq), to_u32(k_input_strides.stride_head), k_scale_cols);
                break;
            case 128:
                compute_k_block_amax_kernel<T, 128><<<grid_dbg_k, block_dbg_k, 0, stream>>>(
                    k_quant_src, k_amax_dev.get(), kv_len,
                    to_u32(k_input_strides.stride_batch), to_u32(k_input_strides.stride_seq), to_u32(k_input_strides.stride_head), k_scale_cols);
                break;
        }
        CUDA_CHECK(cudaGetLastError());
        std::vector<float> host_k_amax(k_scale_elems);
        std::vector<float> host_k_scale(k_scale_elems);
        CUDA_CHECK(cudaMemcpyAsync(host_k_amax.data(), k_amax_dev.get(), k_scale_elems * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(host_k_scale.data(), k_scale.get(), k_scale_elems * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        float max_diff_k = 0.0f;
        size_t max_idx_k = 0;
        for (size_t i = 0; i < k_scale_elems; ++i) {
            const float actual_scale = host_k_amax[i] / 127.0f;
            const float diff = fabsf(actual_scale - host_k_scale[i]);
            if (diff > max_diff_k) {
                max_diff_k = diff;
                max_idx_k = i;
            }
        }
        const int cols_k = k_scale_cols;
        const int heads_total_k = num_k_heads;
        const int batch_idx_k = max_idx_k / (heads_total_k * cols_k);
        const int head_idx_k = (max_idx_k / cols_k) % heads_total_k;
        const int block_idx_k = max_idx_k % cols_k;
        const int token_start_k = block_idx_k * BLKK;
        fprintf(stderr,
                "SAGE_K_AMAX max_diff=%g batch=%d head=%d block=%d token_start=%d actual=%g scale=%g\n",
                max_diff_k,
                batch_idx_k,
                head_idx_k,
                block_idx_k,
                token_start_k,
                host_k_amax[max_idx_k],
                host_k_scale[max_idx_k]);
        if (dump_prefix) {
            sage_debug_dump(dump_prefix, "k_amax", k_amax_dev.get(), k_scale_elems * sizeof(float), stream);
        }
    }

    const bool fused_blocked = dump_prefix != nullptr || debug_quant_host;
    bool enable_v_fused = sage_force_v_fused();
    if (!enable_v_fused && !fused_blocked && !force_host_permute) {
        enable_v_fused = true;
    }
    if (fused_blocked || force_host_permute) {
        enable_v_fused = false;
    }
    if (sage_disable_v_fused()) {
        enable_v_fused = false;
    }

    const bool need_v_transposed = !enable_v_fused || dump_prefix != nullptr || debug_v_compare || force_host_permute;
    ggml_cuda_pool_alloc<T> v_transposed;
    T * v_transposed_ptr = nullptr;
    if (need_v_transposed) {
        v_transposed.alloc(pool, (size_t) batch * num_k_heads * head_dim_padded * kv_len_padded);
        v_transposed_ptr = v_transposed.get();
        switch (head_dim_padded) {
            case 64:
                if (force_host_permute) {
                    GGML_ASSERT(v_padded_ptr != nullptr);
                    transpose_pad_permute_host<T, 64>(v_padded_ptr, v_transposed_ptr, batch, num_k_heads, kv_len, kv_len_padded, stream);
                } else {
                    transpose_pad_permute_strided<T, 64>(
                        static_cast<const T *>(v->data),
                        v_strides,
                        v_dims.head_dim,
                        batch,
                        num_k_heads,
                        kv_len,
                        kv_len_padded,
                        v_transposed_ptr,
                        stream);
                }
                break;
            case 128:
                if (force_host_permute) {
                    GGML_ASSERT(v_padded_ptr != nullptr);
                    transpose_pad_permute_host<T, 128>(v_padded_ptr, v_transposed_ptr, batch, num_k_heads, kv_len, kv_len_padded, stream);
                } else {
                    transpose_pad_permute_strided<T, 128>(
                        static_cast<const T *>(v->data),
                        v_strides,
                        v_dims.head_dim,
                        batch,
                        num_k_heads,
                        kv_len,
                        kv_len_padded,
                        v_transposed_ptr,
                        stream);
                }
                break;
        }
        if (dump_prefix) {
            sage_debug_dump(dump_prefix, "v_transposed", v_transposed_ptr,
                (size_t) batch * num_k_heads * head_dim_padded * kv_len_padded * sizeof(T), stream);
        }
    }

    bool smooth_v = smooth_v_param;
    const bool use_pv_fp16_accum = (pv_accum == GGML_SAGE_PV_ACCUM_FP32_FP16);
    const float v_scale_max = use_pv_fp16_accum ? FP8_SCALE_MAX_FP16 : FP8_SCALE_MAX_FP32;

    const value_strides v_quant_strides = make_value_strides(num_k_heads, head_dim_padded, kv_len_padded);
    const uint32_t scale_stride_head = to_u32(head_dim_padded);
    const uint32_t scale_stride_batch = to_u32((int64_t) num_k_heads * head_dim_padded);
    const uint32_t mean_stride_head = smooth_v ? scale_stride_head : 0;
    const uint32_t mean_stride_batch = smooth_v ? scale_stride_batch : 0;

    const size_t v_fp8_elems = (size_t) batch * num_k_heads * head_dim_padded * kv_len_padded;
    const size_t v_scale_elems = (size_t) batch * num_k_heads * head_dim_padded;

    ggml_cuda_pool_alloc<int8_t> v_fp8(pool, v_fp8_elems);
    ggml_cuda_pool_alloc<float>  v_scale(pool, v_scale_elems);
    float * v_mean_ptr = nullptr;
    std::unique_ptr<ggml_cuda_pool_alloc<float>> v_mean_storage;
    if (smooth_v) {
        v_mean_storage = std::make_unique<ggml_cuda_pool_alloc<float>>(pool, (size_t) batch * num_k_heads * head_dim_padded);
        v_mean_ptr = v_mean_storage->get();
    }

    switch (head_dim_padded) {
        case 64:
            if (enable_v_fused) {
                const T * fusion_src = need_v_transposed ? v_transposed_ptr : v_bhsd_ptr;
                const tensor_strides & fusion_strides = need_v_transposed ? v_transposed_strides : v_bhsd_input_strides;
                if (smooth_v) {
                    quantize_v_per_channel_fused<T, 64, true>(
                        fusion_src,
                        v_fp8.get(),
                        v_mean_ptr,
                        v_scale.get(),
                        v_scale_max,
                        batch,
                        num_k_heads,
                        v_head_dim,
                        kv_len,
                        kv_len_padded,
                        fusion_strides,
                        v_quant_strides,
                        mean_stride_batch,
                        mean_stride_head,
                        scale_stride_batch,
                        scale_stride_head,
                        need_v_transposed,
                        stream);
                } else {
                    quantize_v_per_channel_fused<T, 64, false>(
                        fusion_src,
                        v_fp8.get(),
                        nullptr,
                        v_scale.get(),
                        v_scale_max,
                        batch,
                        num_k_heads,
                        v_head_dim,
                        kv_len,
                        kv_len_padded,
                        fusion_strides,
                        v_quant_strides,
                        mean_stride_batch,
                        mean_stride_head,
                        scale_stride_batch,
                        scale_stride_head,
                        need_v_transposed,
                        stream);
                }
            } else {
                GGML_ASSERT(v_transposed_ptr);
                if (smooth_v) {
                    quantize_v_per_channel<T, true>(v_transposed_ptr, v_fp8.get(), v_mean_ptr, v_scale.get(),
                        v_scale_max, batch, num_k_heads, head_dim_padded, kv_len, kv_len_padded, stream);
                } else {
                    quantize_v_per_channel<T, false>(v_transposed_ptr, v_fp8.get(), nullptr, v_scale.get(),
                        v_scale_max, batch, num_k_heads, head_dim_padded, kv_len, kv_len_padded, stream);
                }
            }
            break;
        case 128:
            if (enable_v_fused) {
                const T * fusion_src = need_v_transposed ? v_transposed_ptr : v_bhsd_ptr;
                const tensor_strides & fusion_strides = need_v_transposed ? v_transposed_strides : v_bhsd_input_strides;
                if (smooth_v) {
                    quantize_v_per_channel_fused<T, 128, true>(
                        fusion_src,
                        v_fp8.get(),
                        v_mean_ptr,
                        v_scale.get(),
                        v_scale_max,
                        batch,
                        num_k_heads,
                        v_head_dim,
                        kv_len,
                        kv_len_padded,
                        fusion_strides,
                        v_quant_strides,
                        mean_stride_batch,
                        mean_stride_head,
                        scale_stride_batch,
                        scale_stride_head,
                        need_v_transposed,
                        stream);
                } else {
                    quantize_v_per_channel_fused<T, 128, false>(
                        fusion_src,
                        v_fp8.get(),
                        nullptr,
                        v_scale.get(),
                        v_scale_max,
                        batch,
                        num_k_heads,
                        v_head_dim,
                        kv_len,
                        kv_len_padded,
                        fusion_strides,
                        v_quant_strides,
                        mean_stride_batch,
                        mean_stride_head,
                        scale_stride_batch,
                        scale_stride_head,
                        need_v_transposed,
                        stream);
                }
            } else {
                GGML_ASSERT(v_transposed_ptr);
                if (smooth_v) {
                    quantize_v_per_channel<T, true>(v_transposed_ptr, v_fp8.get(), v_mean_ptr, v_scale.get(),
                        v_scale_max, batch, num_k_heads, head_dim_padded, kv_len, kv_len_padded, stream);
                } else {
                    quantize_v_per_channel<T, false>(v_transposed_ptr, v_fp8.get(), nullptr, v_scale.get(),
                        v_scale_max, batch, num_k_heads, head_dim_padded, kv_len, kv_len_padded, stream);
                }
            }
            break;
    }
    if (dump_prefix) {
        sage_debug_dump(dump_prefix, "v_fp8", v_fp8.get(),
            v_fp8_elems * sizeof(int8_t), stream);
        sage_debug_dump(dump_prefix, "v_scale", v_scale.get(),
            v_scale_elems * sizeof(float), stream);
        if (smooth_v && v_mean_ptr) {
            sage_debug_dump(dump_prefix, "v_mean", v_mean_ptr,
                (size_t) batch * num_k_heads * head_dim_padded * sizeof(float), stream);
        }
    }

    if (enable_v_fused && debug_v_compare && v_transposed_ptr != nullptr) {
        ggml_cuda_pool_alloc<int8_t> v_fp8_ref(pool, v_fp8_elems);
        ggml_cuda_pool_alloc<float>  v_scale_ref(pool, v_scale_elems);
        if (smooth_v) {
            quantize_v_per_channel<T, true>(
                v_transposed_ptr,
                v_fp8_ref.get(),
                v_mean_ptr,
                v_scale_ref.get(),
                v_scale_max,
                batch,
                num_k_heads,
                head_dim_padded,
                kv_len,
                kv_len_padded,
                stream);
        } else {
            quantize_v_per_channel<T, false>(
                v_transposed_ptr,
                v_fp8_ref.get(),
                nullptr,
                v_scale_ref.get(),
                v_scale_max,
                batch,
                num_k_heads,
                head_dim_padded,
                kv_len,
                kv_len_padded,
                stream);
        }
        CUDA_CHECK(cudaGetLastError());

        const size_t batch_sz = (size_t) batch;
        const size_t heads_sz = (size_t) num_k_heads;

        std::vector<int8_t> host_fp8_fused(v_fp8_elems);
        std::vector<int8_t> host_fp8_ref(v_fp8_elems);
        std::vector<float> host_scale_fused(v_scale_elems);
        std::vector<float> host_scale_ref(v_scale_elems);
        CUDA_CHECK(cudaMemcpyAsync(host_fp8_fused.data(), v_fp8.get(), v_fp8_elems * sizeof(int8_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(host_fp8_ref.data(), v_fp8_ref.get(), v_fp8_elems * sizeof(int8_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(host_scale_fused.data(), v_scale.get(), v_scale_elems * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(host_scale_ref.data(), v_scale_ref.get(), v_scale_elems * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        int payload_diff_max = 0;
        size_t payload_diff_idx = 0;
        float scale_diff_max = 0.0f;
        size_t scale_diff_idx = 0;
        size_t nan_scale_count = 0;
        for (size_t i = 0; i < v_fp8_elems; ++i) {
            const int diff = std::abs((int) host_fp8_fused[i] - (int) host_fp8_ref[i]);
            if (diff > payload_diff_max) {
                payload_diff_max = diff;
                payload_diff_idx = i;
            }
        }
        for (size_t i = 0; i < v_scale_elems; ++i) {
            const float fused_val = host_scale_fused[i];
            if (std::isnan(fused_val)) {
                nan_scale_count++;
                continue;
            }
            const float diff = fabsf(fused_val - host_scale_ref[i]);
            if (diff > scale_diff_max) {
                scale_diff_max = diff;
                scale_diff_idx = i;
            }
        }
        size_t tmp_idx = payload_diff_idx;
        const size_t payload_token = tmp_idx % kv_len_padded;
        tmp_idx /= kv_len_padded;
        const size_t payload_dim = tmp_idx % head_dim_padded;
        tmp_idx /= head_dim_padded;
        const size_t payload_head = tmp_idx % heads_sz;
        const size_t payload_batch = tmp_idx / heads_sz;

        tmp_idx = scale_diff_idx;
        const size_t scale_dim = tmp_idx % head_dim_padded;
        tmp_idx /= head_dim_padded;
        const size_t scale_head = tmp_idx % heads_sz;
        const size_t scale_batch = tmp_idx / heads_sz;

        fprintf(stderr,
                "SAGE_V_FUSED_DEBUG payload_diff=%d b=%zu h=%zu d=%zu t=%zu fused=%d ref=%d | "
                "scale_diff=%g b=%zu h=%zu d=%zu fused=%g ref=%g | nan_scales=%zu/%zu\n",
                payload_diff_max,
                payload_batch,
                payload_head,
                payload_dim,
                payload_token,
                (int) host_fp8_fused[payload_diff_idx],
                (int) host_fp8_ref[payload_diff_idx],
                scale_diff_max,
                scale_batch,
                scale_head,
                scale_dim,
                scale_batch < batch_sz && scale_head < heads_sz ? host_scale_fused[scale_diff_idx] : 0.0f,
                scale_batch < batch_sz && scale_head < heads_sz ? host_scale_ref[scale_diff_idx] : 0.0f,
                nan_scale_count,
                v_scale_elems);
    }

    ggml_cuda_pool_alloc<T> out_padded(pool, (size_t) head_dim_padded * qo_len * num_q_heads * batch);
    const tensor_dims out_pad_dims = { head_dim_padded, qo_len, num_q_heads, batch };
    const tensor_strides out_pad_strides = make_contiguous_strides(out_pad_dims);
    const tensor_dims dst_dims = { head_dim, qo_len, num_q_heads, batch };
    const tensor_strides dst_strides = get_tensor_strides(dst);

    const float sm_scale = softmax_scale;

    switch (head_dim_padded) {
        case 64:
            if (is_causal) {
                launch_sage_kernel<T, 64, MaskMode::kCausal>(
                    q_int8.get(), k_int8.get(), v_fp8.get(), out_padded.get(),
                    q_scale.get(), k_scale.get(), v_scale.get(), v_mean_ptr,
                    batch, num_q_heads, num_k_heads, qo_len, kv_len, num_kv_groups,
                    q_int8_strides, k_int8_strides, out_pad_strides, v_quant_strides,
                    sm_scale, stream, granularity, pv_accum, q_kernel_debug);
            } else {
                launch_sage_kernel<T, 64, MaskMode::kNone>(
                    q_int8.get(), k_int8.get(), v_fp8.get(), out_padded.get(),
                    q_scale.get(), k_scale.get(), v_scale.get(), v_mean_ptr,
                    batch, num_q_heads, num_k_heads, qo_len, kv_len, num_kv_groups,
                    q_int8_strides, k_int8_strides, out_pad_strides, v_quant_strides,
                    sm_scale, stream, granularity, pv_accum, q_kernel_debug);
            }
            break;
        case 128:
            if (is_causal) {
                launch_sage_kernel<T, 128, MaskMode::kCausal>(
                    q_int8.get(), k_int8.get(), v_fp8.get(), out_padded.get(),
                    q_scale.get(), k_scale.get(), v_scale.get(), v_mean_ptr,
                    batch, num_q_heads, num_k_heads, qo_len, kv_len, num_kv_groups,
                    q_int8_strides, k_int8_strides, out_pad_strides, v_quant_strides,
                    sm_scale, stream, granularity, pv_accum, q_kernel_debug);
            } else {
                launch_sage_kernel<T, 128, MaskMode::kNone>(
                    q_int8.get(), k_int8.get(), v_fp8.get(), out_padded.get(),
                    q_scale.get(), k_scale.get(), v_scale.get(), v_mean_ptr,
                    batch, num_q_heads, num_k_heads, qo_len, kv_len, num_kv_groups,
                    q_int8_strides, k_int8_strides, out_pad_strides, v_quant_strides,
                    sm_scale, stream, granularity, pv_accum, q_kernel_debug);
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
    if (dump_prefix) {
        const size_t out_bytes = (size_t) dst_dims.head_dim * dst_dims.seq_len * dst_dims.heads * dst_dims.batch * sizeof(T);
        sage_debug_dump(dump_prefix, "out", static_cast<T *>(dst->data), out_bytes, stream);
    }

    if (debug_kernel && q_kernel_debug_slots > 0 && q_kernel_debug.token_start != nullptr) {
        std::vector<uint32_t> host_q_tokens(q_kernel_debug_slots);
        std::vector<uint32_t> host_q_scale_idx(q_kernel_debug_slots);
        std::vector<float> host_q_scale_val(q_kernel_debug_slots);
        std::vector<uint32_t> host_q_k_scale_idx;
        std::vector<float> host_q_k_scale_val;
        std::vector<float> host_q_dequant;
        std::vector<float> host_q_sm_scale;
        std::vector<float> host_iter_k_scale;
        std::vector<float> host_iter_dequant;
        std::vector<float> host_iter_sm_scale;
        CUDA_CHECK(cudaMemcpyAsync(host_q_tokens.data(), q_kernel_debug.token_start, q_kernel_debug_slots * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(host_q_scale_idx.data(), q_kernel_debug.scale_index, q_kernel_debug_slots * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(host_q_scale_val.data(), q_kernel_debug.scale_value, q_kernel_debug_slots * sizeof(float), cudaMemcpyDeviceToHost, stream));
        if (q_kernel_debug.k_scale_index) {
            host_q_k_scale_idx.resize(q_kernel_debug_slots);
            CUDA_CHECK(cudaMemcpyAsync(host_q_k_scale_idx.data(), q_kernel_debug.k_scale_index, q_kernel_debug_slots * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
        }
        if (q_kernel_debug.k_scale_value) {
            host_q_k_scale_val.resize(q_kernel_debug_slots);
            CUDA_CHECK(cudaMemcpyAsync(host_q_k_scale_val.data(), q_kernel_debug.k_scale_value, q_kernel_debug_slots * sizeof(float), cudaMemcpyDeviceToHost, stream));
        }
        if (q_kernel_debug.dequant_scale) {
            host_q_dequant.resize(q_kernel_debug_slots);
            CUDA_CHECK(cudaMemcpyAsync(host_q_dequant.data(), q_kernel_debug.dequant_scale, q_kernel_debug_slots * sizeof(float), cudaMemcpyDeviceToHost, stream));
        }
        if (q_kernel_debug.sm_scale_value) {
            host_q_sm_scale.resize(q_kernel_debug_slots);
            CUDA_CHECK(cudaMemcpyAsync(host_q_sm_scale.data(), q_kernel_debug.sm_scale_value, q_kernel_debug_slots * sizeof(float), cudaMemcpyDeviceToHost, stream));
        }
        const bool have_iter_debug = debug_kernel_iters &&
            q_kernel_debug.iter_stride != 0 && q_kernel_debug.iter_count != 0;
        if (have_iter_debug) {
            const size_t iter_slots = q_kernel_debug_slots * (size_t) q_kernel_debug.iter_count;
            if (q_kernel_debug.k_scale_iter) {
                host_iter_k_scale.resize(iter_slots);
                CUDA_CHECK(cudaMemcpyAsync(host_iter_k_scale.data(), q_kernel_debug.k_scale_iter, iter_slots * sizeof(float), cudaMemcpyDeviceToHost, stream));
            }
            if (q_kernel_debug.dequant_iter) {
                host_iter_dequant.resize(iter_slots);
                CUDA_CHECK(cudaMemcpyAsync(host_iter_dequant.data(), q_kernel_debug.dequant_iter, iter_slots * sizeof(float), cudaMemcpyDeviceToHost, stream));
            }
            if (q_kernel_debug.sm_scale_iter) {
                host_iter_sm_scale.resize(iter_slots);
                CUDA_CHECK(cudaMemcpyAsync(host_iter_sm_scale.data(), q_kernel_debug.sm_scale_iter, iter_slots * sizeof(float), cudaMemcpyDeviceToHost, stream));
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const int warps_per_block = CTA_Q / WARP_Q;
        const int kernel_print_limit = getenv("GGML_SAGE_KERNEL_PRINT_LIMIT") ? atoi(getenv("GGML_SAGE_KERNEL_PRINT_LIMIT")) : 8;
        fprintf(stderr,
                "SAGE_KERNEL_Q_DEBUG summary: print_limit=%d stride=%d slots=%zu heads=%d iter_count=%u\n",
                kernel_print_limit,
                q_kernel_debug.stride,
                q_kernel_debug_slots,
                num_q_heads,
                have_iter_debug ? q_kernel_debug.iter_count : 0u);
        size_t printed = 0;
        for (int b = 0; b < batch && printed < (size_t) kernel_print_limit; ++b) {
            for (int h = 0; h < num_q_heads && printed < (size_t) kernel_print_limit; ++h) {
                for (int slot = 0; slot < q_kernel_debug.stride && printed < (size_t) kernel_print_limit; ++slot) {
                    const size_t idx = ((size_t) b * num_q_heads + h) * q_kernel_debug.stride + slot;
                    const int block128 = slot / warps_per_block;
                    const int warp = slot % warps_per_block;
                    const int expected_token = block128 * CTA_Q + warp * WARP_Q;
                    const size_t expected_scale_idx =
                        (size_t) b * num_q_heads * q_scale_cols +
                        (size_t) h * q_scale_cols +
                        (size_t) block128 * warps_per_block + warp;
                    const uint32_t recorded_token = host_q_tokens[idx];
                    const uint32_t recorded_scale_idx = host_q_scale_idx[idx];
                    const float recorded_scale = host_q_scale_val[idx];
                    const uint32_t recorded_k_scale_idx = host_q_k_scale_idx.empty() ? 0 : host_q_k_scale_idx[idx];
                    const float recorded_k_scale = host_q_k_scale_val.empty() ? 0.0f : host_q_k_scale_val[idx];
                    const float recorded_dequant = host_q_dequant.empty() ? 0.0f : host_q_dequant[idx];
                    const float recorded_sm = host_q_sm_scale.empty() ? 0.0f : host_q_sm_scale[idx];
                    auto iter_value = [&](const std::vector<float> & vec, int iter_idx) -> float {
                        if (vec.empty() || q_kernel_debug.iter_stride == 0 || q_kernel_debug.iter_count == 0 || iter_idx >= (int) q_kernel_debug.iter_count) {
                            return 0.0f;
                        }
                        const size_t iter_region = (size_t) q_kernel_debug.iter_stride * q_kernel_debug.iter_count;
                        const size_t base = ((size_t) b * num_q_heads + h) * iter_region + slot;
                        return vec[base + (size_t) iter_idx * q_kernel_debug.iter_stride];
                    };
                    fprintf(stderr,
                            "SAGE_KERNEL_Q_DEBUG batch=%d head=%d slot=%d token=%u expected_token=%d "
                            "q_scale_idx=%u expected_q_scale_idx=%zu q_scale=%g "
                            "k_scale_idx=%u k_scale=%g dequant=%g sm_scale=%g",
                            b, h, slot, recorded_token, expected_token,
                            recorded_scale_idx, expected_scale_idx,
                            recorded_scale,
                            recorded_k_scale_idx, recorded_k_scale,
                            recorded_dequant, recorded_sm);
                    if (have_iter_debug &&
                            (!host_iter_k_scale.empty() || !host_iter_dequant.empty() || !host_iter_sm_scale.empty())) {
                        const int iter_print = std::min<int>(q_kernel_debug.iter_count, 4);
                        fprintf(stderr, " | iters");
                        for (int it = 0; it < iter_print; ++it) {
                            fprintf(stderr, " [%d]k=%g dq=%g sm=%g",
                                    it,
                                    iter_value(host_iter_k_scale, it),
                                    iter_value(host_iter_dequant, it),
                                    iter_value(host_iter_sm_scale, it));
                        }
                        if ((int) q_kernel_debug.iter_count > iter_print) {
                            fprintf(stderr, " ...");
                        }
                    }
                    fprintf(stderr, "\n");
                    printed++;
                    if (printed >= (size_t) kernel_print_limit) {
                        break;
                    }
                }
            }
        }
    }
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

    const int32_t pv_mode = ggml_get_op_params_i32(dst, 4);
    if (pv_mode < GGML_SAGE_PV_ACCUM_FP32 || pv_mode > GGML_SAGE_PV_ACCUM_FP32_FP16) {
        return false;
    }

    const int32_t quant_gran = ggml_get_op_params_i32(dst, 5);
    if (quant_gran != GGML_SAGE_QK_GRANULARITY_PER_WARP &&
        quant_gran != GGML_SAGE_QK_GRANULARITY_PER_THREAD) {
        return false;
    }

    return true;
}
#define SAGE_DEBUG 1
