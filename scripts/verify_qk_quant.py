#!/usr/bin/env python3
"""
Compare GGML's dumped Q/K INT8 payloads and scales against the official
SageAttention CUDA quantizers.

Usage:
    uv run python scripts/verify_qk_quant.py \
        --case-prefix tmp/ggml_vs_sage/b1_causal_seq128_d64_seed0 \
        --dump-prefix tmp/qkv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from sageattention.quant import per_warp_int8, per_block_int8


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
    return meta


def load_tensor(prefix: Path, suffix: str, dtype, shape) -> np.ndarray:
    data = np.fromfile(prefix.with_suffix(f".{suffix}.bin"), dtype=dtype)
    if data.size != np.prod(shape):
        raise ValueError(f"{suffix}: expected {np.prod(shape)} values, got {data.size}")
    return data.reshape(shape).copy()


def qkv_from_case(case_prefix: Path, meta: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    head_dim = meta["head_dim"]
    seq_q = meta["seq_q"]
    seq_k = meta["seq_k"]
    num_q = meta["num_q_heads"]
    num_k = meta["num_k_heads"]
    batch = meta["batch"]

    q = load_tensor(case_prefix, "q", np.float16, (head_dim, seq_q, num_q, batch))
    k = load_tensor(case_prefix, "k", np.float16, (head_dim, seq_k, num_k, batch))

    # convert DSHB -> BHSD (tensor_layout="HND")
    q = torch.from_numpy(q).permute(3, 2, 1, 0).contiguous().cuda()
    k = torch.from_numpy(k).permute(3, 2, 1, 0).contiguous().cuda()
    return q, k


def load_int8_dump(prefix: Path, name: str, head_dim: int, seq: int, heads: int, batch: int) -> np.ndarray:
    path = prefix.with_suffix(f".{name}.bin")
    arr = np.fromfile(path, dtype=np.int8)
    expected = head_dim * seq * heads * batch
    if arr.size != expected:
        raise ValueError(f"{path}: expected {expected}, got {arr.size}")
    return arr.reshape(batch, heads, seq, head_dim)


def load_scale(prefix: Path, name: str, shape) -> np.ndarray:
    path = prefix.with_suffix(f".{name}.bin")
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size != np.prod(shape):
        raise ValueError(f"{path}: expected {np.prod(shape)}, got {arr.size}")
    return arr.reshape(shape)


def main():
    parser = argparse.ArgumentParser(description="Verify GGML Q/K INT8 dumps against SageAttention.")
    parser.add_argument("--case-prefix", required=True, type=Path)
    parser.add_argument("--dump-prefix", required=True, type=Path)
    args = parser.parse_args()

    meta = read_meta(args.case_prefix.with_suffix(".meta"))
    smooth_k = bool(meta.get("smooth_k", 1))
    head_dim = meta["head_dim"]
    head_dim_padded = 64 if head_dim <= 64 else 128
    seq_q = meta["seq_q"]
    seq_k = meta["seq_k"]
    num_q = meta["num_q_heads"]
    num_k = meta["num_k_heads"]
    batch = meta["batch"]

    q, k = qkv_from_case(args.case_prefix, meta)
    with torch.inference_mode():
        q_int8_ref, q_scale_ref, k_int8_ref, k_scale_ref = per_warp_int8(
            q.clone(),
            k.clone(),
            tensor_layout="HND",
            km=k.mean(dim=2, keepdim=True) if smooth_k else None,
        )

    # reshape to numpy (batch, head, seq, dim)
    q_int8_ref = q_int8_ref.cpu().numpy()
    q_scale_ref = q_scale_ref.cpu().numpy()
    k_int8_ref = k_int8_ref.cpu().numpy()
    k_scale_ref = k_scale_ref.cpu().numpy()

    dump_prefix = args.dump_prefix
    q_int8_dump = load_int8_dump(dump_prefix, "q_int8", head_dim_padded, seq_q, num_q, batch)
    q_scale_cols = ((seq_q + 128 - 1) // 128) * (128 // 32)
    q_scale_dump = load_scale(dump_prefix, "q_scale", (batch, num_q, q_scale_cols))

    k_int8_dump = load_int8_dump(dump_prefix, "k_int8", head_dim_padded, seq_k, num_k, batch)
    k_scale_dump = load_scale(dump_prefix, "k_scale", (batch, num_k, (seq_k + 64 - 1) // 64))

    def padded_view(arr, pad_dim, pad_seq=None):
        out = np.zeros((batch, arr.shape[1], arr.shape[2], pad_dim), dtype=arr.dtype)
        out[:, :, :, :head_dim] = arr
        if pad_seq is not None and pad_seq > arr.shape[2]:
            tmp = np.zeros((batch, arr.shape[1], pad_seq, pad_dim), dtype=arr.dtype)
            tmp[:, :, :arr.shape[2], :arr.shape[3]] = out
            return tmp
        return out

    q_int8_ref_pad = padded_view(q_int8_ref, head_dim_padded)
    k_int8_ref_pad = padded_view(k_int8_ref, head_dim_padded)

    q_scale_ref_flat = q_scale_ref
    k_scale_ref_flat = k_scale_ref

    q_scale_diff = np.max(np.abs(q_scale_dump - q_scale_ref_flat))
    k_scale_diff = np.max(np.abs(k_scale_dump - k_scale_ref_flat))
    q_payload_diff = np.max(np.abs(q_int8_dump - q_int8_ref_pad))
    k_payload_diff = np.max(np.abs(k_int8_dump - k_int8_ref_pad))

    print(f"[q_scale] max diff: {q_scale_diff}")
    print(f"[k_scale] max diff: {k_scale_diff}")
    print(f"[q_int8] max diff: {q_payload_diff}")
    print(f"[k_int8] max diff: {k_payload_diff}")


if __name__ == "__main__":
    main()
