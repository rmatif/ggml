# SageAttention Layout Bug – Root Cause & Fix

## What Broke

SageAttention expects tensors in **BHSd** order (`[batch, head, seq, head_dim]`) once they are padded.  
GGML stores tensors as **DSHB** (`[dim, seq, head, batch]`) and relied on a CUDA helper to reorder them. Two issues made our kernel dumps diverge from the official Sage implementation:

1. **`convert_dshb_to_bhsd_kernel` wrote the wrong order** – it copied elements directly to the destination buffer without swapping the sequence/head axes. Any consumer that assumed BHSd (our quantizers, transpose, Python harness) was silently reading the wrong values.
2. **Values (`V`) skipped the layout conversion when padding wasn’t required** – the transpose/permute kernel sometimes read raw DSHB memory, so the shared-memory tiles contained the wrong tokens even though padding and permutation looked correct.

Because of these bugs, the Python harness (`scripts/run_sage_kernel.py`) compared Sage’s CUDA kernel (fed with correctly-resized tensors) against a reference dump that still used the dim-major layout. The harness reported the infamous ~0.39 RMS “mismatch”, even though the actual kernel math was fine.

## The Fix

1. **Correct the GPU DSHB→BHSd conversion** (`src/ggml-cuda/sage-attn.cu:600`):
   ```cpp
   const size_t dst_index =
       (((size_t) b * heads + h) * seq_len + s) * head_dim + d;
   const size_t src_index =
       (((size_t) d * seq_len + s) * heads + h) * batch + b;
   dst[dst_index] = src[src_index];
   ```
   This matches the host reference implementation byte-for-byte.

2. **Always convert V tensors before the transpose/permute** (`sage_attn_impl`):
   - Pad `V` when either the head dimension or the sequence length requires it *or* when dumps/debugging are enabled.
   - Run `convert_dshb_to_bhsd_{host,device}` to produce a contiguous BHSd buffer and feed that buffer to the transpose kernel, regardless of padding.

3. **Make the Python harness reshape in the proper order**:
   - `scripts/run_sage_kernel.py` now loads Q/K/FP8 dumps with row-major order (matching the new layout) and explicitly makes tensors contiguous before launching the official kernel.
   - `scripts/verify_qk_quant.py` / `scripts/verify_v_quant.py` reshape dumps in row-major order as well so the reference quantizers see the same bytes GGML produced.

With these changes every stage—quantization, FP8 transpose, and the fused kernel—consumes the same layout SageAttention uses.

## How to Verify Locally

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j16

# Grab a saved case and dump Q/K/V + outputs.
GGML_CUDA_FORCE_VMM=1 \
GGML_SAGE_DUMP_QKV=tmp/kernel_dump \
GGML_SAGE_DUMP_OUT=tmp/outdump \
build/bin/test-sageflash --load tmp/ggml_vs_sage_regen/b1_causal_seq128_d64_seed0 \
    --compare sage --iters 1

# Quantization parity
uv run python scripts/verify_qk_quant.py \
    --case-prefix tmp/ggml_vs_sage_regen/b1_causal_seq128_d64_seed0 \
    --dump-prefix tmp/kernel_dump
uv run python scripts/verify_v_quant.py \
    --case-prefix tmp/ggml_vs_sage_regen/b1_causal_seq128_d64_seed0 \
    --dump-prefix tmp/kernel_dump

# Kernel parity against official SageAttention
uv run python scripts/run_sage_kernel.py \
    --dump-prefix tmp/kernel_dump \
    --case-prefix tmp/ggml_vs_sage_regen/b1_causal_seq128_d64_seed0 \
    --head-dim 64 --seq-q 128 --seq-k 128 \
    --num-q-heads 12 --num-k-heads 6 --batch 1 \
    --causal --granularity per_warp --pv-accum fp32+fp16 \
    --compare-dump tmp/outdump
```

All three commands should now report `max diff: 0` and `official vs reference rms: 0`, confirming GGML’s dumps are byte-identical to SageAttention’s.
