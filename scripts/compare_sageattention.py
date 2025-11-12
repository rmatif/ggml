import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import sageattention
from sageattention.quant import per_warp_int8 as per_warp_int8_cuda, per_channel_fp8
from sageattention.triton.quant_per_thread import per_thread_int8 as per_thread_int8_triton

@dataclass
class Case:
    name: str
    head_dim: int
    seq_q: int
    seq_k: int
    num_q_heads: int
    num_k_heads: int
    batch: int
    is_causal: bool
    smooth_k: bool
    granularity: str

CASES = [
    Case("b2_seq256_d128", 128, 256, 256, 16, 8, 2, False, True, "per_warp"),
    Case("b1_causal_seq128_d64", 64, 128, 128, 12, 6, 1, True, True, "per_warp"),
    Case("b1_causal_seq128_d64_thread", 64, 128, 128, 12, 6, 1, True, True, "per_thread"),
    Case("b2_seq256_d128_thread", 128, 256, 256, 16, 8, 2, False, True, "per_thread"),
]

def run_case(case: Case, seed: int, smooth_v: bool, pv_accum: str):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    dtype = torch.float16

    q = torch.randn(case.batch, case.num_q_heads, case.seq_q, case.head_dim, device=device, dtype=dtype)
    k = torch.randn(case.batch, case.num_k_heads, case.seq_k, case.head_dim, device=device, dtype=dtype)
    v = torch.randn(case.batch, case.num_k_heads, case.seq_k, case.head_dim, device=device, dtype=dtype)

    sm_scale = 1.0 / math.sqrt(case.head_dim)

    with torch.no_grad():
        out_sage = sageattention.sageattn_qk_int8_pv_fp8_cuda(
            q.clone(),
            k.clone(),
            v.clone(),
            tensor_layout="HND",
            is_causal=case.is_causal,
            qk_quant_gran=case.granularity,
            sm_scale=sm_scale,
            pv_accum_dtype=pv_accum,
            smooth_k=case.smooth_k,
            smooth_v=smooth_v,
            return_lse=False,
        )

        if case.num_q_heads != case.num_k_heads:
            assert case.num_q_heads % case.num_k_heads == 0
            repeat = case.num_q_heads // case.num_k_heads
            k_flash = k.repeat_interleave(repeat, dim=1)
            v_flash = v.repeat_interleave(repeat, dim=1)
        else:
            k_flash = k
            v_flash = v

        def flatten_heads(t):
            b, h, l, d = t.shape
            return t.reshape(b * h, l, d)

        q_flat = flatten_heads(q)
        k_flat = flatten_heads(k_flash)
        v_flat = flatten_heads(v_flash)

        flash = torch.nn.functional.scaled_dot_product_attention(
            q_flat, k_flat, v_flat, attn_mask=None, dropout_p=0.0, is_causal=case.is_causal
        )
        out_flash = flash.reshape(case.batch, case.num_q_heads, case.seq_q, case.head_dim)

    diff = (out_sage - out_flash).float()
    abs_diff = diff.abs()
    max_diff = abs_diff.max().item()
    max_index = abs_diff.argmax().item()
    rms = diff.pow(2).mean().sqrt().item()
    sage_val = out_sage.reshape(-1)[max_index].item()
    flash_val = out_flash.reshape(-1)[max_index].item()
    return max_diff, rms, max_index, sage_val, flash_val, q, k, v, out_sage, out_flash


def dump_case(prefix: str, case: Case, q, k, v, out_sage, out_flash, smooth_v: bool, pv_accum: str):
    base = Path(prefix)
    base.parent.mkdir(parents=True, exist_ok=True)
    meta_lines = [
        f"head_dim {case.head_dim}",
        f"seq_q {case.seq_q}",
        f"seq_k {case.seq_k}",
        f"num_q_heads {case.num_q_heads}",
        f"num_k_heads {case.num_k_heads}",
        f"batch {case.batch}",
        f"is_causal {int(case.is_causal)}",
        f"smooth_k {int(case.smooth_k)}",
        f"granularity {case.granularity}",
    ]
    (base.with_suffix(".meta")).write_text("\n".join(meta_lines) + "\n")

    def save_tensor(name: str, tensor: torch.Tensor, dtype: torch.dtype):
        arr = tensor.permute(3, 2, 1, 0).contiguous().to(dtype).cpu().numpy()
        np.asarray(arr).tofile(f"{prefix}.{name}.bin")

    save_tensor("q", q, torch.float16)
    save_tensor("k", k, torch.float16)
    save_tensor("v", v, torch.float16)
    save_tensor("sage", out_sage, torch.float32)
    save_tensor("flash", out_flash, torch.float32)
    print(f"  dumped tensors to {prefix}.*")


def main():
    parser = argparse.ArgumentParser(description="Compare SageAttention vs Torch SDPA")
    parser.add_argument("--case", choices=[c.name for c in CASES], help="Case to run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smooth-v", action="store_true", help="Enable V smoothing when supported")
    parser.add_argument(
        "--pv-accum",
        default="fp32+fp16",
        choices=["fp32", "fp32+fp32", "fp32+fp16"],
        help="SageAttention pv accumulation mode",
    )
    parser.add_argument("--dump-prefix", help="Optional prefix to dump q/k/v and outputs for ggml", default=None)
    args = parser.parse_args()

    case = next(c for c in CASES if c.name == args.case)
    max_diff, rms, idx, sage_val, flash_val, q, k, v, out_sage, out_flash = run_case(
        case, args.seed, args.smooth_v, args.pv_accum
    )

    total = case.batch * case.num_q_heads * case.seq_q * case.head_dim
    print(f"Case {case.name}")
    print(f"  settings: gran={case.granularity} smooth_k={case.smooth_k} smooth_v={args.smooth_v} pv_accum={args.pv_accum}")
    print(f"  dims: batch={case.batch} seq_q={case.seq_q} seq_k={case.seq_k} nq={case.num_q_heads} nk={case.num_k_heads} d={case.head_dim}")
    print(f"  metrics: max_diff={max_diff:.6e} rms={rms:.6e}")
    print(f"  worst index: flat={idx}/{total} sage={sage_val:.6e} flash={flash_val:.6e}")

    if args.dump_prefix:
        dump_case(args.dump_prefix, case, q, k, v, out_sage, out_flash, args.smooth_v, args.pv_accum)


if __name__ == "__main__":
    main()
