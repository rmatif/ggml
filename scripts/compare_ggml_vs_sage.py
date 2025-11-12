#!/usr/bin/env python3
"""
Accuracy harness comparing GGML's SageAttention kernel against the official
`sageattention` package.

Workflow:
  1. Use SageAttention's PyTorch wrapper to generate random Q/K/V tensors and
     the reference Sage output (same cases as compare_sageattention.py).
  2. Dump tensors to disk (meta + q/k/v/reference files) so ggml's
     `test-sageflash --load` path can consume them.
  3. Invoke the ggml harness with `--compare sage` to compute RMS/max diffs
     directly against the official reference.

Example:
    uv run python scripts/compare_ggml_vs_sage.py \
        --ggml-bin build/bin/test-sageflash \
        --case b1_causal_seq128_d64 --iters 3 --pv-accum fp32+fp16
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch

from compare_sageattention import CASES, Case, dump_case, run_case

DIFF_RE = re.compile(r"diff.*max ([0-9eE\+\-\.]+) rms_avg ([0-9eE\+\-\.]+)")


@dataclass
class HarnessResult:
    seed: int
    max_diff: float
    rms_diff: float
    ggml_ms: float
    flash_ms: float
    raw_output: str


def parse_case(name: str) -> Case:
    for case in CASES:
        if case.name == name:
            return case
    raise ValueError(f"unknown case '{name}'")


def run_ggml_case(ggml_bin: Path, prefix: Path) -> HarnessResult:
    env = os.environ.copy()
    cmd = [
        str(ggml_bin),
        "--load",
        str(prefix),
        "--compare",
        "sage",
        "--iters",
        "1",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"test-sageflash failed (code={proc.returncode})\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

    max_diff = float("nan")
    rms_diff = float("nan")
    last_stdout = proc.stdout.strip().splitlines()
    for line in last_stdout[::-1]:
        match = DIFF_RE.search(line)
        if match:
            max_diff = float(match.group(1))
            rms_diff = float(match.group(2))
            break

    # Extract timings from the block (`timings (avg): sage X ms, flash Y ms`)
    ggml_ms = float("nan")
    flash_ms = float("nan")
    for line in last_stdout:
        if "timings (avg)" in line:
            parts = line.replace(",", "").split()
            try:
                gg_idx = parts.index("sage")
                ggml_ms = float(parts[gg_idx + 1])
                fl_idx = parts.index("flash")
                flash_ms = float(parts[fl_idx + 1])
            except (ValueError, IndexError):
                pass
            break

    return HarnessResult(
        seed=0,
        max_diff=max_diff,
        rms_diff=rms_diff,
        ggml_ms=ggml_ms,
        flash_ms=flash_ms,
        raw_output=proc.stdout,
    )


def ensure_clean(prefix: Path) -> None:
    for suffix in [".meta", ".q.bin", ".k.bin", ".v.bin", ".sage.bin", ".flash.bin"]:
        path = prefix.with_suffix(suffix)
        if path.exists():
            path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GGML SageAttention vs official SageAttention outputs.")
    parser.add_argument("--ggml-bin", default="build/bin/test-sageflash", help="Path to ggml test-sageflash executable.")
    parser.add_argument("--case", required=True, choices=[c.name for c in CASES], help="Test case to run.")
    parser.add_argument("--iters", type=int, default=1, help="Number of iterations (unique seeds).")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    parser.add_argument("--smooth-v", action="store_true", help="Enable V smoothing when generating reference tensors.")
    parser.add_argument(
        "--pv-accum",
        default="fp32+fp16",
        choices=["fp32", "fp32+fp32", "fp32+fp16"],
        help="SageAttention pv accumulation mode.",
    )
    parser.add_argument("--tmp-dir", default="tmp/ggml_vs_sage", help="Directory for intermediate dumps.")
    parser.add_argument("--keep-dumps", action="store_true", help="Keep intermediate tensor dumps.")
    args = parser.parse_args()

    case = parse_case(args.case)
    ggml_bin = Path(args.ggml_bin)
    if not ggml_bin.exists():
        raise FileNotFoundError(f"ggml binary not found: {ggml_bin}")

    tmp_root = Path(args.tmp_dir)
    tmp_root.mkdir(parents=True, exist_ok=True)

    results: List[HarnessResult] = []

    for it in range(args.iters):
        seed = args.seed + it
        prefix = tmp_root / f"{case.name}_seed{seed}"
        ensure_clean(prefix)

        _, _, _, _, _, q, k, v, out_sage, out_flash = run_case(case, seed, args.smooth_v, args.pv_accum)
        dump_case(str(prefix), case, q, k, v, out_sage, out_flash, args.smooth_v, args.pv_accum)

        harness_res = run_ggml_case(ggml_bin, prefix)
        harness_res.seed = seed
        results.append(harness_res)

        print(harness_res.raw_output)

        if not args.keep_dumps:
            ensure_clean(prefix)

    if results:
        max_rms = max(r.rms_diff for r in results)
        avg_rms = sum(r.rms_diff for r in results) / len(results)
        max_diff = max(r.max_diff for r in results)
        print("==== Summary ====")
        print(f"Case: {case.name}")
        print(f"Iterations: {len(results)}")
        print(f"RMS diff (avg): {avg_rms:.6e}, worst: {max_rms:.6e}")
        print(f"Max diff overall: {max_diff:.6e}")


if __name__ == "__main__":
    main()
