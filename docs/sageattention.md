# SageAttention SM89 Backend

GGML ships a CUDA implementation of SageAttention SM89 (`GGML_OP_SAGE_ATTN_SM89_FP16`).
This backend targets Hopper/Lovelace GPUs and mirrors the public SageAttention
API (granularity, smoothing, PV accumulation). Use this document as the
reference for tuning options, debug tooling, and accuracy harnesses.

## User-Facing Knobs

All knobs can be set per `ggml_sage_attn_sm89_fp16()` invocation and overridden
globally through environment variables. The CPU op stores the following fields in
`op_params` and the CUDA backend honors them verbatim:

| Parameter | CLI / Env | Description |
|-----------|-----------|-------------|
| `smooth_k` | `--smooth-k/--no-smooth-k`, `GGML_SAGE_SMOOTH_K=0|1` | Enable K smoothing (recommended for per-warp granularity). |
| `smooth_v` | `--smooth-v/--no-smooth-v`, `GGML_SAGE_SMOOTH_V=0|1` | Enable V smoothing. Requires `pv_accum=fp32`; incompatible modes are automatically disabled with a runtime warning. |
| `pv_accum` (`fp32`, `fp32+fp32`, `fp32+fp16`) | `--pv-accum ...`, `GGML_SAGE_PV_ACCUM=...` | Matches SageAttention’s PV accumulation dtype selection. Controls whether the kernel materializes the PV intermediate buffer and whether FP8 scales are capped at 2.25 or 448. |
| `quant_granularity` | `--granularity per_warp|per_thread` | Select Q/K quantization granularity. Mixed-head configurations are supported. |

Tips:

- The environment overrides apply to *all* subsequent `ggml_sage_attn_sm89_fp16`
  calls inside the process. Each override logs once so downstream apps know
  why their requested settings changed.
- `smooth_v` silently degrades accuracy when paired with non-`fp32` PV modes.
  The runtime enforces SageAttention’s contract by forcing `smooth_v=0` and
  emitting a warning.

## Feature Coverage (SM89)

This port mirrors SageAttention’s SM89 FP8 path:

- Per-warp and per-thread Q/K quantizers with optional K smoothing.
- All PV accumulation dtypes exposed upstream (`fp32`, `fp32+fp32`, `fp32+fp16`),
  matching FP8 scale limits (448 vs 2.25) and V smoothing requirements.
- Hopper/Ada head sizes 64 and 128; tensors are padded/permutted so ggml’s native
  layout feeds the CUDA kernels directly.
- Debug hooks and dumps to inspect head-dim padding, quantized payloads, and V
  transpose/FP8 conversion.

Not yet implemented (and therefore intentionally unsupported):

- Returning log-sum-exp buffers (`return_lse=True`).
- Alternative tensor layouts (e.g. `tensor_layout="NHD"`).
- FP16 PV kernels (`sageattn_qk_int8_pv_fp16_cuda`) or SM80/SM90 backends.
- Variable-length sequences or external masks beyond the causal flag.

If you need these features, they must be ported separately.

## C API usage

```c
ggml_tensor * ggml_sage_attn_sm89_fp16(
        ggml_context * ctx,
        ggml_tensor  * q,
        ggml_tensor  * k,
        ggml_tensor  * v,
        float          softmax_scale,
        bool           is_causal,
        bool           smooth_k,
        bool           smooth_v,
        enum ggml_sage_pv_accum pv_mode,
        enum ggml_sage_qk_granularity gran);
```

Call it inside your ggml graph, then cast to F32 (if needed) before comparing
with flash attention or SageAttention dumps. The op enforces tensor shape/type
constraints and records the knob selections in `op_params` so the CUDA backend
can read them.

## Accuracy & Debug Harness

Two Python helpers complement `tests/test-sageflash`:

1. `scripts/compare_sageattention.py` – generates random Q/K/V tensors, runs the
   official SageAttention (`sageattention` PyPI package), runs Torch SDPA for the
   flash reference, and dumps tensors to `${prefix}.{q,k,v,sage,flash}.bin`. The
   `.meta` file now records `smooth_k`, `smooth_v`, `pv_accum`, and granularity.
2. `scripts/compare_ggml_vs_sage.py` – reads the dumps, uses
   `test-sageflash --load ... --compare sage`, and reports RMS / max errors plus
   timings. This script requires the extra `.sage_quant.bin` emitted by the first
   helper; without it you will observe the historical “0.39 RMS” false alarm.
   That spike does **not** indicate an accuracy regression—it simply means the
   run is comparing quantized GGML outputs against Sage’s *float* tensors. As
   soon as the `.sage_quant.bin` file is present (or regenerated), RMS drops to
   ~1e-4.

### Legacy dumps & the “0.39 RMS” phantom

Older dumps only stored `*.sage.bin` (float layout) which does **not** match the
quantized kernel output. When those files are loaded, ggml converts them to BHSd
order and prints a warning:

```
[test-sageflash] warning: <prefix> missing .sage_quant.bin; converting legacy layout
(expect ~0.39 RMS vs float refs if not regenerated)
```

If you see that line—or a similar warning from `compare_ggml_vs_sage.py`—you can
ignore the high RMS; regenerate the dumps with `compare_sageattention.py` to get
bitwise comparisons again. Do **not** chase this as a “new accuracy issue.”

Recommended workflow:

```bash
uv run python scripts/compare_sageattention.py \
    --case b1_causal_seq128_d64 \
    --pv-accum fp32+fp16 \
    --dump-prefix tmp/ggml_vs_sage/b1_causal_seq128_d64

uv run python scripts/compare_ggml_vs_sage.py \
    --case b1_causal_seq128_d64 \
    --ggml-bin build/bin/test-sageflash \
    --tmp-dir tmp/ggml_vs_sage \
    --keep-dumps
```

`tests/test-sageflash` exposes matching CLI flags (`--smooth-k`, `--smooth-v`,
`--pv-accum`, `--granularity`, `--compare flash|sage`). Set
`GGML_SAGE_DUMP=/tmp/prefix` to capture intermediate tensors (padded BHSd buffers,
int8 payloads, FP8 V tiles, scales). Additional env vars toggle instrumentation;
see below.

## Debug toggles

| Variable | Description |
|----------|-------------|
| `GGML_SAGE_DUMP=<prefix>` | Dump every intermediate tensor (padded Q/K/V, int8 payloads, FP8 V tiles, scales, means, outputs). |
| `GGML_SAGE_DEBUG_QUANT`, `_HOST`, `_DEV` | Enable per-warp diagnostics for the CUDA and host quantizers. Useful when verifying scale indexing. |
| `GGML_SAGE_FORCE_SIMPLE_QK`, `GGML_SAGE_FORCE_HOST_QK` | Swap in simplified or CPU quantizers to isolate accuracy issues. |
| `GGML_SAGE_Q_SERIAL` | Serializes quantization across heads to diagnose stride mismatches. |
| `GGML_SAGE_DUMP_QK_INT8` | Writes the quantized Q/K payloads alongside their scale buffers. |
| `GGML_SAGE_DEBUG_KERNEL_ITERS`, `GGML_SAGE_KERNEL_PRINT_LIMIT` | Instrument the SM89 kernel with per-iteration telemetry. |
| `GGML_SAGE_SAMPLE_PREFIX`, `GGML_SAGE_SAMPLE_HEAD/BLOCK/WARP/LIMIT` | Restrict dumps to selected heads/blocks/warps. |

Unset every variable for production; the defaults match SageAttention’s behavior.

## Tests

- `tests/test-backend-ops --filter sage_attn` keeps coverage inside ggml’s unit
  suite. Shapes span both granularities, causal/non-causal, and every PV mode.
- `tests/test-sageflash` exercises larger workloads and can load deterministic
  input/output dumps via `--load <prefix>`. Combine with `--compare sage` to
  measure accuracy against official SageAttention.
- `scripts/compare_sageattention.py` + `scripts/compare_ggml_vs_sage.py` automate
  the Sage vs ggml diff workflow (see above). Always keep the `.sage_quant.bin`
  files produced by the first script to avoid the 0.39 RMS phantom.

Run the CUDA build and the harness via:

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j16
build/bin/test-sageflash --case b1_causal_seq128_d64 --compare sage --iters 1
```

The same executable compares against flash attention by default. Toggle the
environment overrides to evaluate different knob combinations without modifying
downstream application code.
