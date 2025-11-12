#!/usr/bin/env python3
"""
Run SageAttention's official SM89 int8/fp8 kernel on dumped quantized tensors.

Usage:
    uv run python scripts/run_sage_kernel.py \
        --dump-prefix tmp/ggml_dump \
        --case-prefix tmp/b1_per_warp \
        --head-dim 64 --seq-q 128 --seq-k 128 \
        --num-q-heads 12 --num-k-heads 6 --batch 1 \
        [--tensor-layout hnd|nhd] [--causal]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import sageattention.sm89_compile as sm89


def load_int8_tensor(path: Path, head_dim: int, seq: int, heads: int, batch: int):
    data = np.fromfile(path, dtype=np.int8)
    expected = head_dim * seq * heads * batch
    if data.size != expected:
        raise ValueError(f"unexpected size for {path} (got {data.size}, expected {expected})")
    data = data.reshape(head_dim, seq, heads, batch)
    tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(data, (3, 2, 1, 0)))).cuda()
    return tensor


def load_scale(path: Path, shape):
    data = np.fromfile(path, dtype=np.float32)
    if data.size != np.prod(shape):
        raise ValueError(f"unexpected size for {path} (got {data.size}, expected {np.prod(shape)})")
    return torch.from_numpy(data.reshape(shape)).cuda()


def load_v_fp8(path: Path, batch: int, heads: int, head_dim: int, seq: int):
    data = np.fromfile(path, dtype=np.int8)
    expected = batch * heads * head_dim * seq
    if data.size != expected:
        raise ValueError(f"unexpected size for {path} (got {data.size}, expected {expected})")
    tensor = torch.from_numpy(data).cuda().reshape(batch, heads, head_dim, seq).contiguous()
    return tensor


def main():
    parser = argparse.ArgumentParser(description="Run SageAttention SM89 kernel on dumped tensors.")
    parser.add_argument("--dump-prefix", required=True, type=Path)
    parser.add_argument("--case-prefix", required=True, type=Path)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--seq-q", type=int, required=True)
    parser.add_argument("--seq-k", type=int, required=True)
    parser.add_argument("--num-q-heads", type=int, required=True)
    parser.add_argument("--num-k-heads", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--tensor-layout", choices=["hnd", "nhd"], default="hnd")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--granularity", choices=["per_warp", "per_thread"], default="per_warp")
    parser.add_argument("--sm-scale", type=float, default=None)
    parser.add_argument("--pv-accum", choices=["fp32", "fp32+fp32", "fp32+fp16"], default="fp32+fp16")
    parser.add_argument("--smooth-v", action="store_true")
    args = parser.parse_args()

    head_dim = args.head_dim
    head_dim_padded = 64 if head_dim <= 64 else 128
    kv_len_pad = ((args.seq_k + 63) // 64) * 64
    tensor_layout_flag = 1 if args.tensor_layout == "hnd" else 0
    q_gran = 2 if args.granularity == "per_warp" else 3

    q_int8 = load_int8_tensor(args.dump_prefix.with_suffix(".q_int8.bin"), head_dim_padded, args.seq_q, args.num_q_heads, args.batch)
    k_int8 = load_int8_tensor(args.dump_prefix.with_suffix(".k_int8.bin"), head_dim_padded, args.seq_k, args.num_k_heads, args.batch)
    v_fp8 = load_v_fp8(args.dump_prefix.with_suffix(".v_fp8.bin"), args.batch, args.num_k_heads, head_dim_padded, kv_len_pad)

    q_blocks = (args.seq_q + 128 - 1) // 128
    warps_per_block = 128 // 32
    q_scale_cols = q_blocks * warps_per_block if args.granularity == "per_warp" else q_blocks * warps_per_block * 8
    q_scale = load_scale(args.dump_prefix.with_suffix(".q_scale.bin"), (args.batch, args.num_q_heads, q_scale_cols))

    k_scale_cols = (args.seq_k + 64 - 1) // 64 if args.granularity == "per_warp" else ((args.seq_k + 64 - 1) // 64) * 4
    k_scale = load_scale(args.dump_prefix.with_suffix(".k_scale.bin"), (args.batch, args.num_k_heads, k_scale_cols))

    v_scale = load_scale(args.dump_prefix.with_suffix(".v_scale.bin"), (args.batch, args.num_k_heads, head_dim_padded))
    v_mean = None
    if args.smooth_v:
        if args.pv_accum != "fp32":
            raise SystemExit("--smooth-v currently supported only with --pv-accum=fp32")
        v_mean = load_scale(args.dump_prefix.with_suffix(".v_mean.bin"), (args.batch, args.num_k_heads, head_dim_padded))

    sm_scale = args.sm_scale
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)

    out = torch.zeros(args.batch, args.num_q_heads, args.seq_q, head_dim_padded, dtype=torch.float16, device="cuda")
    if args.pv_accum == "fp32":
        if args.smooth_v:
            kernel = torch.ops.sageattention_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn
            kernel(q_int8, k_int8, v_fp8, out, q_scale, k_scale, v_scale, v_mean,
                   tensor_layout_flag, 1 if args.causal else 0, q_gran, sm_scale, 0)
        else:
            kernel = torch.ops.sageattention_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn
            kernel(q_int8, k_int8, v_fp8, out, q_scale, k_scale, v_scale,
                   tensor_layout_flag, 1 if args.causal else 0, q_gran, sm_scale, 0)
    elif args.pv_accum == "fp32+fp32":
        kernel = torch.ops.sageattention_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf
        kernel(q_int8, k_int8, v_fp8, out, q_scale, k_scale, v_scale,
               tensor_layout_flag, 1 if args.causal else 0, q_gran, sm_scale, 0)
    else:
        kernel = torch.ops.sageattention_sm89.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf
        kernel(q_int8, k_int8, v_fp8, out, q_scale, k_scale, v_scale,
               tensor_layout_flag, 1 if args.causal else 0, q_gran, sm_scale, 0)

    out = out.cpu().numpy()[..., :head_dim]
    ref = np.fromfile(args.case_prefix.with_suffix(".sage.bin"), dtype=np.float32).reshape(head_dim, args.seq_q, args.num_q_heads, args.batch)
    out = np.transpose(out, (3, 2, 1, 0)).reshape(ref.shape)

    diff = out - ref
    print("official kernel vs reference rms:", np.sqrt(np.mean(diff**2)))
    print("max diff:", np.max(np.abs(diff)))


if __name__ == "__main__":
    main()
