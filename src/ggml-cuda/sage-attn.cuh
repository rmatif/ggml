#pragma once

#include "common.cuh"

void ggml_cuda_sage_attn_sm89_fp16(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_sage_attn_sm89_fp16_supported(int device, const ggml_tensor * dst);
