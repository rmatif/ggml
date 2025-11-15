#!/usr/bin/env python3
"""
Compare GGML's dumped SageAttention outputs against the official SageAttention
reference tensors produced by scripts/compare_sageattention.py.

Usage:
    uv run python scripts/verify_attn_out.py \
        --case-prefix tmp/ggml_vs_sage/b1_causal_seq128_d64_seed0 \
        --dump-prefix tmp/outdump
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


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


def load_dump(path: Path, dtype, shape):
    data = np.fromfile(path, dtype=dtype)
    if data.size != np.prod(shape):
        raise ValueError(f"{path}: expected {np.prod(shape)}, got {data.size}")
    return data.reshape(shape)


def main():
    parser = argparse.ArgumentParser(description="Verify SageAttention kernel outputs.")
    parser.add_argument("--case-prefix", required=True, type=Path, help="Prefix containing *.meta and .sage_quant.bin")
    parser.add_argument("--dump-prefix", required=True, type=Path, help="Prefix used via GGML_SAGE_DUMP_OUT")
    args = parser.parse_args()

    meta = read_meta(args.case_prefix.with_suffix(".meta"))
    head_dim = meta["head_dim"]
    seq_q = meta["seq_q"]
    num_q = meta["num_q_heads"]
    batch = meta["batch"]
    head_dim_padded = 64 if head_dim <= 64 else 128

    # Load GGML dumps
    out_padded = load_dump(args.dump_prefix.with_suffix(".out_padded.bin"),
                           np.float16,
                           (head_dim_padded, seq_q, num_q, batch))
    out_trimmed = load_dump(args.dump_prefix.with_suffix(".out.bin"),
                            np.float16,
                            (head_dim, seq_q, num_q, batch))

    # Reference from SageAttention (saved by compare_sageattention.py)
    ref = np.fromfile(args.case_prefix.with_suffix(".sage_quant.bin"), dtype=np.float32)
    ref = ref.reshape(batch, num_q, seq_q, head_dim)

    # Convert GGML layout (dim, seq, head, batch) -> (batch, head, seq, dim)
    ggml_out = np.transpose(out_trimmed, (3, 2, 1, 0)).astype(np.float32)

    diff = ggml_out - ref
    rms = np.sqrt(np.mean(diff**2))
    max_diff = np.max(np.abs(diff))
    max_idx = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)

    perp = np.transpose(out_padded[..., :seq_q], (3, 2, 1, 0))[:, :, :, :head_dim].astype(np.float32)
    pad_diff = np.sqrt(np.mean((perp - ref) ** 2))

    print(f"[verify] RMS diff: {rms:.6e}")
    print(f"[verify] Max diff: {max_diff:.6e} at (batch,head,seq,dim)={max_idx}")
    print(f"[verify] RMS diff using out_padded slice: {pad_diff:.6e}")


if __name__ == "__main__":
    main()
