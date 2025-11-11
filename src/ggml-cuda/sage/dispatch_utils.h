/*
 * Derived from SageAttention dispatch utilities.
 */

#pragma once

#include <sstream>
#include <stdexcept>

#include "ggml.h"

#define GGML_SAGE_THROW(msg)             \
    do {                                 \
        std::ostringstream oss;          \
        oss << msg;                      \
        throw std::invalid_argument(oss.str()); \
    } while (0)

#define DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, ...)          \
    if ((head_dim) == 64) {                                 \
        constexpr int HEAD_DIM = 64;                        \
        __VA_ARGS__                                         \
    } else if ((head_dim) == 128) {                         \
        constexpr int HEAD_DIM = 128;                       \
        __VA_ARGS__                                         \
    } else {                                                \
        GGML_SAGE_THROW("Unsupported head dim: " << (head_dim)); \
    }

#define DISPATCH_CAUSAL(is_causal, IS_CAUSAL, ...)          \
    if (is_causal) {                                        \
        constexpr bool IS_CAUSAL = true;                    \
        __VA_ARGS__                                         \
    } else {                                                \
        constexpr bool IS_CAUSAL = false;                   \
        __VA_ARGS__                                         \
    }

#define DISPATCH_QK_QUANT_GRAN(gran, GRAN, ...)                      \
    if ((gran) == static_cast<int>(QuantGranularity::kPerWarp)) {    \
        constexpr int GRAN = static_cast<int>(QuantGranularity::kPerWarp); \
        __VA_ARGS__                                                  \
    } else if ((gran) == static_cast<int>(QuantGranularity::kPerThread)) { \
        constexpr int GRAN = static_cast<int>(QuantGranularity::kPerThread); \
        __VA_ARGS__                                                  \
    } else {                                                         \
        GGML_SAGE_THROW("Unsupported qk_quant_gran: " << (gran));    \
    }

#define DISPATCH_RETURN_LSE(rtn_lse, RETURN_LSE, ...)        \
    if (rtn_lse) {                                          \
        constexpr bool RETURN_LSE = true;                   \
        __VA_ARGS__                                         \
    } else {                                                \
        constexpr bool RETURN_LSE = false;                  \
        __VA_ARGS__                                         \
    }

#define DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(dtype, c_type, ...) \
    if ((dtype) == GGML_TYPE_F16) {                              \
        using c_type = half;                                     \
        __VA_ARGS__                                              \
    } else if ((dtype) == GGML_TYPE_BF16) {                      \
        using c_type = nv_bfloat16;                              \
        __VA_ARGS__                                              \
    } else {                                                     \
        GGML_SAGE_THROW("Unsupported data type");                \
    }
