#!/usr/bin/env python3
"""
Helper to validate GGML's SageAttention V transpose + FP8 quantization pipeline.

Given a case prefix (meta + raw tensors) and a GGML dump prefix
(`GGML_SAGE_DUMP=... build/bin/test-sageflash --load ...`), this script:
  1. Loads the original FP16 V tensor (HND layout) and the dumped
     transposed buffer (`v_transposed`).
  2. Reproduces Sage's transpose+pad+permute step entirely on the CPU
     and confirms it matches the GGML dump byte-for-byte.
  3. Runs SageAttention's reference `per_channel_fp8` quantizer
     (with `scale_max=2.25` and smoothing disabled) on the raw tensor
     and checks that the resulting FP8 payload + per-channel scales
     match GGML's `v_fp8` / `v_scale` dumps.

Usage:
    uv run python scripts/verify_v_quant.py \
        --case-prefix tmp/b1_per_warp \
        --dump-prefix tmp/ggml_dump
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from sageattention.quant import per_channel_fp8


def read_meta(meta_path: Path) -> dict:
    meta = {}
    with meta_path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                key, val = parts
                try:
                    meta[key] = int(val)
                except ValueError:
                    meta[key] = val
    required = ["head_dim", "seq_k", "num_k_heads", "batch"]
    missing = [k for k in required if k not in meta]
    if missing:
        raise ValueError(f"Missing keys in meta file {meta_path}: {missing}")
    return meta


def permute_tokens(seq_len: int) -> np.ndarray:
    """Permutation applied by Sage's transpose_pad_permute kernel."""
    idx = np.arange(seq_len, dtype=np.int32)
    low = idx & 0xF
    high = idx & ~0xF
    b0 = low & 1
    b1 = (low >> 1) & 1
    b2 = (low >> 2) & 1
    b3 = (low >> 3) & 1
    new_low = b0 | (b3 << 1) | (b1 << 2) | (b2 << 3)
    return high | new_low


def load_tensor(prefix: Path, suffix: str, dtype, shape) -> np.ndarray:
    data = np.fromfile(prefix.with_suffix(f".{suffix}.bin"), dtype=dtype)
    if data.size != np.prod(shape):
        raise ValueError(f"Unexpected size for {suffix}: got {data.size}, expected {np.prod(shape)}")
    return data.reshape(shape, order="F").copy()


def verify_transpose(case_prefix: Path, dump_prefix: Path, meta: dict) -> float:
    head_dim = meta["head_dim"]
    seq_k = meta["seq_k"]
    num_k_heads = meta["num_k_heads"]
    batch = meta["batch"]
    head_dim_padded = 64 if head_dim <= 64 else 128
    kv_len_padded = ((seq_k + 63) // 64) * 64

    v = load_tensor(case_prefix, "v", np.float16, (head_dim, seq_k, num_k_heads, batch))
    v = np.transpose(v, (3, 2, 1, 0))  # (b, h, seq, dim)

    dump_path = dump_prefix.with_suffix(".v_transposed.bin")
    v_trans_dump = np.fromfile(dump_path, dtype=np.float16)
    v_trans_dump = v_trans_dump.reshape(batch, num_k_heads, head_dim_padded, kv_len_padded)

    padded = np.zeros((batch, num_k_heads, head_dim_padded, kv_len_padded), dtype=np.float16)
    padded[:, :, :head_dim, :seq_k] = np.transpose(v, (0, 1, 3, 2))
    perm = permute_tokens(kv_len_padded)
    v_trans_ref = np.zeros_like(v_trans_dump)
    v_trans_ref[:, :, :, perm] = padded

    diff = np.max(np.abs(v_trans_ref - v_trans_dump))
    print(f"[transpose] max diff: {diff}")
    return float(diff)


def verify_fp8(case_prefix: Path, dump_prefix: Path, meta: dict) -> Tuple[float, float]:
    head_dim = meta["head_dim"]
    seq_k = meta["seq_k"]
    num_k_heads = meta["num_k_heads"]
    batch = meta["batch"]
    head_dim_padded = 64 if head_dim <= 64 else 128
    kv_len_padded = ((seq_k + 63) // 64) * 64
    smooth_v = bool(meta.get("smooth_v", 0))
    pv_accum = meta.get("pv_accum", "fp32+fp16")
    scale_max = 2.25 if pv_accum == "fp32+fp16" else 448.0

    v = load_tensor(case_prefix, "v", np.float16, (head_dim, seq_k, num_k_heads, batch))
    v = np.transpose(v, (3, 2, 1, 0))  # (b, h, seq, dim)
    v_t = torch.from_numpy(v.copy()).cuda()

    with torch.inference_mode():
        v_fp8_ref, v_scale_ref, vm = per_channel_fp8(
            v_t, tensor_layout="HND", scale_max=scale_max, smooth_v=smooth_v
        )

    v_fp8_ref_bytes = v_fp8_ref.view(torch.int8).cpu().numpy()
    v_scale_ref = v_scale_ref.cpu().numpy()

    v_fp8_dump = np.fromfile(dump_prefix.with_suffix(".v_fp8.bin"), dtype=np.int8)
    v_fp8_dump = v_fp8_dump.reshape(batch, num_k_heads, head_dim_padded, kv_len_padded)

    v_scale_dump = np.fromfile(dump_prefix.with_suffix(".v_scale.bin"), dtype=np.float32)
    v_scale_dump = v_scale_dump.reshape(batch, num_k_heads, head_dim_padded)

    v_scale_diff = np.max(np.abs(v_scale_dump[:, :, :head_dim] - v_scale_ref))
    v_fp8_diff = np.max(
        np.abs(v_fp8_dump[:, :, :head_dim, :seq_k] - v_fp8_ref_bytes[:, :, :, :seq_k])
    )

    print(f"[fp8] scale max diff: {v_scale_diff}")
    print(f"[fp8] payload max diff: {v_fp8_diff}")
    v_mean_diff = 0.0
    if smooth_v:
        v_mean_dump = np.fromfile(dump_prefix.with_suffix(".v_mean.bin"), dtype=np.float32)
        v_mean_dump = v_mean_dump.reshape(batch, num_k_heads, head_dim_padded)
        v_mean_ref = vm.cpu().numpy()
        v_mean_diff = np.max(np.abs(v_mean_dump[:, :, :head_dim] - v_mean_ref))
        print(f"[fp8] mean max diff: {v_mean_diff}")
    return float(v_scale_diff), float(v_fp8_diff), float(v_mean_diff)


def main():
    parser = argparse.ArgumentParser(description="Verify SageAttention V FP8 pipeline.")
    parser.add_argument("--case-prefix", required=True, type=Path, help="Prefix of tensors/meta to load.")
    parser.add_argument("--dump-prefix", required=True, type=Path, help="Prefix produced by GGML_SAGE_DUMP.")
    args = parser.parse_args()

    meta = read_meta(args.case_prefix.with_suffix(".meta"))
    transpose_diff = verify_transpose(args.case_prefix, args.dump_prefix, meta)
    scale_diff, fp8_diff, mean_diff = verify_fp8(args.case_prefix, args.dump_prefix, meta)

    if transpose_diff == scale_diff == fp8_diff == mean_diff == 0.0:
        print("All checks passed.")
    else:
        print("Differences detected; see metrics above.")


if __name__ == "__main__":
    main()
