#include "conv-transpose-1d.cuh"

template <typename src0_t, typename src1_t>
static __global__ void conv_transpose_1d_kernel(
        const int s0, const int p0, const int d0, const int output_size,
        const int src0_ne0, const int src0_ne1, const int src0_ne2, const int src0_ne3,
        const int src1_ne0, const int src1_ne1, const int src1_ne2, const int src1_ne3,
        const int dst_ne0, const int dst_ne1, const int dst_ne2, const int dst_ne3,
        const src0_t * src0, const src1_t * src1,  float * dst) {
    int global_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_index >= output_size) {
        return;
    }

    int out_index = global_index / dst_ne0;

    float accumulator = 0.0f;

    const int idx = global_index % dst_ne0;
    const int k = src0_ne0;

    // valid input indices i satisfy: idx >= i*s0 && idx < i*s0 + k
    // => ceil((idx - (k - 1)) / s0) <= i <= floor(idx / s0)
    int i_start = (idx - (k - 1) + s0 - 1) / s0;
    int i_end   = idx / s0;
    if (i_start < 0) i_start = 0;
    if (i_end >= src1_ne0) i_end = src1_ne0 - 1;

    if (i_start <= i_end) {
        for (int c = 0; c < src0_ne2; c++) {
            const int kernel_offset = (src0_ne0 * src0_ne1 * c) + (out_index * src0_ne0);
            const int input_offset  = src1_ne0 * c;

            for (int i = i_start; i <= i_end; i++) {
                const int weight_idx = idx - i * s0;
                const float kernel_weight = (float)src0[kernel_offset + weight_idx];
                const float input_value   = (float)src1[input_offset + i];
                accumulator += kernel_weight * input_value;
            }
        }
    }

    dst[global_index] = accumulator;
    GGML_UNUSED_VARS(p0, d0, src0_ne3, src1_ne3, dst_ne3, src1_ne1, dst_ne1, src1_ne2, dst_ne2);
}

template <typename src0_t, typename src1_t>
static void conv_transpose_1d_f32_f32_cuda(
        const int s0, const int p0, const int d0, const int output_size,
        const int src0_ne0, const int src0_ne1, const int src0_ne2, const int src0_ne3,
        const int src1_ne0, const int src1_ne1, const int src1_ne2, const int src1_ne3,
        const int dst_ne0, const int dst_ne1, const int dst_ne2, const int dst_ne3,
        const src0_t * src0, const src1_t * src1,  float * dst,
        cudaStream_t stream) {

    const int num_blocks = (output_size + CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE - 1) / CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE;
    conv_transpose_1d_kernel<src0_t, src1_t><<<num_blocks,CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE, 0, stream>>>(
        s0,p0,d0,output_size,
        src0_ne0, src0_ne1,  src0_ne2, src0_ne3,
        src1_ne0, src1_ne1,  src1_ne2, src1_ne3,
        dst_ne0,  dst_ne1,   dst_ne2,  dst_ne3,
        src0,src1, dst);
}

template <typename src0_t>
static void conv_transpose_1d_dispatch_src1(
        const int s0, const int p0, const int d0, const int output_size,
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst,
        const src0_t * src0_d, float * dst_d, cudaStream_t stream) {
    switch (src1->type) {
        case GGML_TYPE_F32:
            conv_transpose_1d_f32_f32_cuda<src0_t, float>(s0, p0, d0, output_size,
                src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                src0_d, (const float *)src1->data, dst_d, stream);
            return;
        case GGML_TYPE_F16:
            conv_transpose_1d_f32_f32_cuda<src0_t, half>(s0, p0, d0, output_size,
                src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                src0_d, (const half *)src1->data, dst_d, stream);
            return;
        case GGML_TYPE_BF16:
            conv_transpose_1d_f32_f32_cuda<src0_t, nv_bfloat16>(s0, p0, d0, output_size,
                src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                src0_d, (const nv_bfloat16 *)src1->data, dst_d, stream);
            return;
        default:
            GGML_ABORT("conv_transpose_1d: unsupported src1 type");
    }
}

void ggml_cuda_op_conv_transpose_1d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    const int32_t * opts = (const int32_t *)dst->op_params;

    const int s0 = opts[0];
    const int p0 = 0;//opts[3];
    const int d0 = 1;//opts[4];

    const int64_t output_size = ggml_nelements(dst);

    switch (src0->type) {
        case GGML_TYPE_F32:
            conv_transpose_1d_dispatch_src1<float>(s0, p0, d0, output_size, src0, src1, dst, (const float *)src0->data, dst_d, stream);
            return;
        case GGML_TYPE_F16:
            conv_transpose_1d_dispatch_src1<half>(s0, p0, d0, output_size, src0, src1, dst, (const half *)src0->data, dst_d, stream);
            return;
        case GGML_TYPE_BF16:
            conv_transpose_1d_dispatch_src1<nv_bfloat16>(s0, p0, d0, output_size, src0, src1, dst, (const nv_bfloat16 *)src0->data, dst_d, stream);
            return;
        default:
            GGML_ABORT("conv_transpose_1d: unsupported src0 type");
    }
}
