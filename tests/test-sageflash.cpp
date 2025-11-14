#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <vector>

struct sage_case {
    std::string name;
    int head_dim;
    int seq_q;
    int seq_k;
    int num_q_heads;
    int num_k_heads;
    int batch;
    bool is_causal;
    bool smooth_k;
    ggml_sage_qk_granularity granularity;
    bool smooth_v;
    ggml_sage_pv_accum pv_accum;
};

struct case_result {
    double sage_ms;
    double flash_ms;
    double max_diff;
    double rms_diff;
    size_t max_index;
    float  sage_val;
    float  ref_val;
    std::string reference;
};

struct cli_options {
    std::vector<std::string> case_filters;
    int iters = 1;
    uint64_t seed = 1234;
    bool list_only = false;
    bool verbose = false;
    bool show_help = false;
    std::string load_prefix;
    bool granularity_override_set = false;
    ggml_sage_qk_granularity granularity_override = GGML_SAGE_QK_GRANULARITY_PER_WARP;
    bool smooth_k_override_set = false;
    bool smooth_k_override = false;
    bool smooth_v_override_set = false;
    bool smooth_v_override = false;
    bool pv_accum_override_set = false;
    ggml_sage_pv_accum pv_accum_override = GGML_SAGE_PV_ACCUM_FP32_FP16;
    enum class compare_mode {
        flash,
        sage,
    } compare = compare_mode::flash;
};

struct external_data {
    bool active = false;
    sage_case cfg;
    std::vector<ggml_fp16_t> q;
    std::vector<ggml_fp16_t> k;
    std::vector<ggml_fp16_t> v;
    std::vector<float> ref_sage_quant;
    std::vector<float> ref_sage;
    std::vector<float> ref_flash;
    bool has_ref_sage_quant = false;
    bool has_ref_sage = false;
    bool has_ref_flash = false;
    std::string prefix;
};

static void fill_tensor_uniform(ggml_tensor * tensor, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const int64_t n = ggml_nelements(tensor);

    std::vector<uint8_t> host(ggml_nbytes(tensor));

    switch (tensor->type) {
        case GGML_TYPE_F16: {
            auto * data = reinterpret_cast<ggml_fp16_t *>(host.data());
            for (int64_t i = 0; i < n; ++i) {
                data[i] = ggml_fp32_to_fp16(dist(rng));
            }
        } break;
        case GGML_TYPE_F32: {
            auto * data = reinterpret_cast<float *>(host.data());
            for (int64_t i = 0; i < n; ++i) {
                data[i] = dist(rng);
            }
        } break;
        default:
            GGML_ABORT("test-sageflash: unsupported tensor type");
    }

    ggml_backend_tensor_set(tensor, host.data(), 0, host.size());
}

static bool parse_pv_accum_string(const std::string & val, ggml_sage_pv_accum & out) {
    if (val == "fp32") {
        out = GGML_SAGE_PV_ACCUM_FP32;
        return true;
    }
    if (val == "fp32+fp32") {
        out = GGML_SAGE_PV_ACCUM_FP32_FP32;
        return true;
    }
    if (val == "fp32+fp16") {
        out = GGML_SAGE_PV_ACCUM_FP32_FP16;
        return true;
    }
    return false;
}

static bool read_meta_file(const std::string & path, sage_case & cfg) {
    std::ifstream fin(path);
    if (!fin) {
        fprintf(stderr, "failed to open meta file %s\n", path.c_str());
        return false;
    }
    cfg.name = path;
    std::string key;
    cfg.granularity = GGML_SAGE_QK_GRANULARITY_PER_WARP;
    cfg.smooth_v = false;
    cfg.pv_accum = GGML_SAGE_PV_ACCUM_FP32_FP16;

    while (fin >> key) {
        if (key == "head_dim") {
            fin >> cfg.head_dim;
        } else if (key == "seq_q") {
            fin >> cfg.seq_q;
        } else if (key == "seq_k") {
            fin >> cfg.seq_k;
        } else if (key == "num_q_heads") {
            fin >> cfg.num_q_heads;
        } else if (key == "num_k_heads") {
            fin >> cfg.num_k_heads;
        } else if (key == "batch") {
            fin >> cfg.batch;
        } else if (key == "is_causal") {
            int v; fin >> v; cfg.is_causal = v != 0;
        } else if (key == "smooth_k") {
            int v; fin >> v; cfg.smooth_k = v != 0;
        } else if (key == "smooth_v") {
            int v; fin >> v; cfg.smooth_v = v != 0;
        } else if (key == "pv_accum") {
            std::string val; fin >> val;
            if (!parse_pv_accum_string(val, cfg.pv_accum)) {
                fprintf(stderr, "unknown pv_accum '%s' in %s\n", val.c_str(), path.c_str());
                return false;
            }
        } else if (key == "granularity") {
            std::string val; fin >> val;
            if (val == "per_thread") {
                cfg.granularity = GGML_SAGE_QK_GRANULARITY_PER_THREAD;
            } else {
                cfg.granularity = GGML_SAGE_QK_GRANULARITY_PER_WARP;
            }
        }
    }
    return true;
}

template<typename T>
static bool load_binary(const std::string & path, std::vector<T> & data, size_t expected_elems) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "failed to open %s\n", path.c_str());
        return false;
    }
    fin.seekg(0, std::ios::end);
    size_t bytes = fin.tellg();
    fin.seekg(0, std::ios::beg);
    size_t elems = bytes / sizeof(T);
    if (expected_elems && elems != expected_elems) {
        fprintf(stderr, "unexpected size in %s (got %zu expected %zu)\n", path.c_str(), elems, expected_elems);
        return false;
    }
    data.resize(elems);
    fin.read(reinterpret_cast<char *>(data.data()), bytes);
    return true;
}

// Legacy SageAttention dumps store tensors with the fastest-moving dimension
// being `dim`, followed by seq, head, batch (DSHB). GGML (and the rest of
// this harness) expects `[batch, head, seq, dim]`. The difference silently
// produces ~0.39 RMS errors that look like kernel bugs, so keep the conversion
// and be vocal whenever we have to fall back to it.
static void convert_dsbh_to_bhsd(std::vector<float> & data, int head_dim, int seq, int heads, int batch) {
    if (data.empty()) {
        return;
    }
    std::vector<float> reordered(data.size());
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            for (int s = 0; s < seq; ++s) {
                for (int d = 0; d < head_dim; ++d) {
                    const size_t dst =
                        (((size_t) b * heads + h) * seq + s) * head_dim + d;
                    const size_t src =
                        ((((size_t) d * seq) + s) * heads + h) * batch + b;
                    reordered[dst] = data[src];
                }
            }
        }
    }
    data.swap(reordered);
}

static bool load_external_case(const std::string & prefix, external_data & ext) {
    ext.prefix = prefix;
    sage_case cfg = {};
    if (!read_meta_file(prefix + ".meta", cfg)) {
        return false;
    }
    if (cfg.head_dim == 0) {
        fprintf(stderr, "invalid metadata in %s.meta\n", prefix.c_str());
        return false;
    }
    const size_t q_elems = (size_t) cfg.head_dim * cfg.seq_q * cfg.num_q_heads * cfg.batch;
    const size_t k_elems = (size_t) cfg.head_dim * cfg.seq_k * cfg.num_k_heads * cfg.batch;
    const size_t v_elems = k_elems;
    const size_t out_elems = (size_t) cfg.head_dim * cfg.seq_q * cfg.num_q_heads * cfg.batch;

    if (!load_binary(prefix + ".q.bin", ext.q, q_elems)) return false;
    if (!load_binary(prefix + ".k.bin", ext.k, k_elems)) return false;
    if (!load_binary(prefix + ".v.bin", ext.v, v_elems)) return false;
    ext.has_ref_sage_quant = load_binary(prefix + ".sage_quant.bin", ext.ref_sage_quant, out_elems);
    ext.has_ref_sage = load_binary(prefix + ".sage.bin", ext.ref_sage, out_elems);
    ext.has_ref_flash = load_binary(prefix + ".flash.bin", ext.ref_flash, out_elems);
    if (ext.has_ref_sage_quant) {
        // already stored in BHSd order
    } else if (ext.has_ref_sage) {
        fprintf(stderr,
                "[test-sageflash] warning: %s missing .sage_quant.bin; converting legacy layout "
                "(expect ~0.39 RMS vs float refs if not regenerated; this is NOT an accuracy bug — "
                "regenerate dumps via compare_sageattention.py)\n",
                prefix.c_str());
        convert_dsbh_to_bhsd(ext.ref_sage, cfg.head_dim, cfg.seq_q, cfg.num_q_heads, cfg.batch);
    }
    if (ext.has_ref_flash) {
        convert_dsbh_to_bhsd(ext.ref_flash, cfg.head_dim, cfg.seq_q, cfg.num_q_heads, cfg.batch);
    }

    ext.cfg = cfg;
    ext.active = true;
    return true;
}

static double run_graph(ggml_context * ctx, ggml_backend_t backend, ggml_tensor * root, std::vector<float> & output) {
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, root);

    const int64_t t0 = ggml_time_us();
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
    const int64_t t1 = ggml_time_us();

    output.resize(ggml_nelements(root));
    ggml_backend_tensor_get(root, output.data(), 0, output.size() * sizeof(float));

    return (t1 - t0)/1000.0;
}

static size_t estimate_ctx_mem(const sage_case & cfg) {
    const int64_t max_seq = std::max(cfg.seq_q, cfg.seq_k);
    const int64_t max_heads = std::max(cfg.num_q_heads, cfg.num_k_heads);
    const size_t elems = (size_t) cfg.head_dim * max_seq * max_heads * cfg.batch;
    size_t bytes = elems * sizeof(float) * 16; // generous safety factor for intermediate tensors
    const size_t min_bytes = 64ull * 1024ull * 1024ull;
    bytes = std::max(bytes, min_bytes);
    // round up to GGML_MEM_ALIGN boundary
    const size_t align = GGML_MEM_ALIGN;
    bytes = ((bytes + align - 1) / align) * align;
    return bytes;
}

static void tensor_set_from_vec(ggml_tensor * tensor, const std::vector<ggml_fp16_t> & data) {
    GGML_ASSERT(ggml_nbytes(tensor) == data.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(tensor, data.data(), 0, data.size() * sizeof(ggml_fp16_t));
}

static case_result run_case(const sage_case & cfg, ggml_backend_t backend, uint64_t seed, const external_data * ext = nullptr,
        std::vector<float> * sage_dump = nullptr, std::vector<float> * flash_dump = nullptr,
        cli_options::compare_mode compare = cli_options::compare_mode::flash) {
    std::mt19937 rng(seed);
    const bool use_loaded = ext && ext->active;

    ggml_init_params params = {
        /* .mem_size   = */ estimate_ctx_mem(cfg),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };

    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx);

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, cfg.head_dim, cfg.seq_q, cfg.num_q_heads, cfg.batch);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, cfg.head_dim, cfg.seq_k, cfg.num_k_heads, cfg.batch);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, cfg.head_dim, cfg.seq_k, cfg.num_k_heads, cfg.batch);

    const float scale = 1.0f/std::sqrt((float) cfg.head_dim);

    ggml_tensor * sage = ggml_sage_attn_sm89_fp16(ctx, q, k, v, scale, cfg.is_causal, cfg.smooth_k, cfg.smooth_v, cfg.pv_accum, cfg.granularity);
    ggml_tensor * sage_f32 = ggml_cast(ctx, sage, GGML_TYPE_F32);

    ggml_tensor * qf = ggml_cast(ctx, q, GGML_TYPE_F32);
    ggml_tensor * kf = ggml_cast(ctx, k, GGML_TYPE_F32);
    ggml_tensor * vf = ggml_cast(ctx, v, GGML_TYPE_F32);
    ggml_tensor * flash = ggml_flash_attn_ext(ctx, qf, kf, vf, nullptr, scale, 0.0f, 0.0f);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer);

    if (use_loaded) {
        tensor_set_from_vec(q, ext->q);
        tensor_set_from_vec(k, ext->k);
        tensor_set_from_vec(v, ext->v);
    } else {
        fill_tensor_uniform(q, rng);
        fill_tensor_uniform(k, rng);
        fill_tensor_uniform(v, rng);
    }

    std::vector<float> sage_out;
    std::vector<float> flash_out;

    case_result res = {};
    res.reference = (compare == cli_options::compare_mode::flash) ? "flash" : "sageattention";
    res.ref_val = std::numeric_limits<float>::quiet_NaN();
    res.sage_val = std::numeric_limits<float>::quiet_NaN();
    res.sage_ms = run_graph(ctx, backend, sage_f32, sage_out);
    res.flash_ms = run_graph(ctx, backend, flash, flash_out);
    res.max_index = 0;
    res.sage_val = 0.0f;

    auto has_non_finite = [](const std::vector<float> & vals) {
        for (float v : vals) {
            if (!std::isfinite(v)) {
                return true;
            }
        }
        return false;
    };

    bool ref_has_non_finite = false;
    const std::vector<float> * ref_output = nullptr;
    if (compare == cli_options::compare_mode::flash) {
        ref_output = &flash_out;
        ref_has_non_finite = has_non_finite(*ref_output);
    } else {
        GGML_ASSERT(ext);
        if (ext->has_ref_sage_quant) {
            ref_output = &ext->ref_sage_quant;
        } else {
            GGML_ASSERT(ext->has_ref_sage);
            ref_output = &ext->ref_sage;
        }
        ref_has_non_finite = has_non_finite(*ref_output);
    }

    if (has_non_finite(sage_out) || ref_has_non_finite) {
        const bool sage_bad = has_non_finite(sage_out);
        const bool ref_bad = ref_has_non_finite;
        printf("WARNING[%s]: non-finite outputs detected (sage=%d, flash=%d)\n",
               cfg.name.c_str(),
               sage_bad ? 1 : 0,
               ref_bad ? 1 : 0);
        if (sage_bad) {
            for (size_t i = 0; i < sage_out.size(); ++i) {
                if (!std::isfinite(sage_out[i])) {
                    printf("  first bad sage value @%zu = %f\n", i, sage_out[i]);
                    break;
                }
            }
        }
        res.max_diff = std::numeric_limits<double>::infinity();
        res.rms_diff = std::numeric_limits<double>::quiet_NaN();
    } else {
        double sum_sq = 0.0;
        double max_abs = 0.0;
        size_t max_idx = 0;
        for (size_t i = 0; i < sage_out.size(); ++i) {
            const double diff = (double) sage_out[i] - (double) (*ref_output)[i];
            const double abs_diff = std::abs(diff);
            if (abs_diff > max_abs) {
                max_abs = abs_diff;
                max_idx = i;
                res.sage_val = sage_out[i];
                res.ref_val = (*ref_output)[i];
            }
            sum_sq += diff*diff;
        }
        res.max_diff = max_abs;
        res.max_index = max_idx;
        res.rms_diff = std::sqrt(sum_sq / sage_out.size());
        res.ref_val = (*ref_output)[max_idx];
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);

    if (sage_dump) {
        *sage_dump = sage_out;
    }
    if (flash_dump) {
        *flash_dump = flash_out;
    }

    return res;
}

static case_result run_case_iters(const sage_case & cfg, ggml_backend_t backend, uint64_t seed_base, int iters) {
    GGML_ASSERT(iters > 0);
    case_result agg = {};
    agg.reference = "flash";
    double sage_sum = 0.0;
    double flash_sum = 0.0;
    double rms_sum = 0.0;
    double max_diff = 0.0;
    bool max_init = false;

    for (int it = 0; it < iters; ++it) {
        const uint64_t seed = seed_base + (uint64_t) it * 10007ull;
        const case_result cur = run_case(cfg, backend, seed, nullptr, nullptr, nullptr, cli_options::compare_mode::flash);
        sage_sum += cur.sage_ms;
        flash_sum += cur.flash_ms;
        rms_sum += cur.rms_diff;
        if (!max_init || cur.max_diff > max_diff) {
            max_diff = cur.max_diff;
            agg.max_index = cur.max_index;
            agg.sage_val = cur.sage_val;
            agg.ref_val = cur.ref_val;
            agg.reference = cur.reference;
            max_init = true;
        }
    }

    agg.sage_ms = sage_sum / iters;
    agg.flash_ms = flash_sum / iters;
    agg.rms_diff = rms_sum / iters;
    agg.max_diff = max_diff;
    return agg;
}

static const char * granularity_name(ggml_sage_qk_granularity gran) {
    switch (gran) {
        case GGML_SAGE_QK_GRANULARITY_PER_WARP:   return "per_warp";
        case GGML_SAGE_QK_GRANULARITY_PER_THREAD: return "per_thread";
        default:                                  return "unknown";
    }
}

static const char * pv_accum_name(ggml_sage_pv_accum pv) {
    switch (pv) {
        case GGML_SAGE_PV_ACCUM_FP32:        return "fp32";
        case GGML_SAGE_PV_ACCUM_FP32_FP32:   return "fp32+fp32";
        case GGML_SAGE_PV_ACCUM_FP32_FP16:   return "fp32+fp16";
        default:                             return "unknown";
    }
}

static void print_case_report(const sage_case & cfg, const case_result & res, int iters, bool verbose) {
    const double speedup = (res.sage_ms > 0.0) ? (res.flash_ms / res.sage_ms) : 0.0;
    printf("\nCase %s\n", cfg.name.c_str());
    printf("  shape: batch=%d seq_q=%d seq_k=%d nq=%d nk=%d d=%d causal=%d smooth_k=%d smooth_v=%d pv=%s\n",
           cfg.batch, cfg.seq_q, cfg.seq_k, cfg.num_q_heads, cfg.num_k_heads, cfg.head_dim,
           cfg.is_causal ? 1 : 0, cfg.smooth_k ? 1 : 0,
           cfg.smooth_v ? 1 : 0, pv_accum_name(cfg.pv_accum));
    printf("  granularity: %s\n", granularity_name(cfg.granularity));
    printf("  iterations: %d\n", iters);
    printf("  timings (avg): sage %.3f ms, flash %.3f ms (speedup %.2fx)\n", res.sage_ms, res.flash_ms, speedup);
    printf("  reference: %s\n", res.reference.c_str());
    printf("  diff (%s): max %.3e rms_avg %.3e\n", res.reference.c_str(), res.max_diff, res.rms_diff);
    if (verbose && std::isfinite(res.max_diff)) {
        size_t idx = res.max_index;
        const int64_t dim = cfg.head_dim;
        const int64_t seq = cfg.seq_q;
        const int64_t heads = cfg.num_q_heads;
        const int64_t batch = cfg.batch;
        int64_t d0 = idx % dim;
        idx /= dim;
        int64_t d1 = idx % seq;
        idx /= seq;
        int64_t d2 = idx % heads;
        idx /= heads;
        int64_t d3 = idx;
        printf("  worst element: batch=%lld head=%lld seq=%lld dim=%lld (sage=%e ref=%e)\n",
               (long long) d3,
               (long long) d2,
               (long long) d1,
               (long long) d0,
               (double) res.sage_val,
               (double) res.ref_val);
    }
}

static void print_usage(const char * prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --case <name>       Run only the specified case (can be repeated).\n");
    printf("  --cases a,b,c       Comma-separated list of case names to run.\n");
    printf("  --iters <n>         Number of iterations per case (default 1).\n");
    printf("  --seed <value>      Base RNG seed (default 1234).\n");
    printf("  --list              List available case names and exit.\n");
    printf("  --verbose           Print additional per-case diagnostics (worst element info).\n");
    printf("  --load <prefix>     Load tensors from files generated by compare_sageattention.py.\n");
    printf("  --granularity per_warp|per_thread  Override Sage quantization granularity.\n");
    printf("  --smooth-k / --no-smooth-k  Override key smoothing flag.\n");
    printf("  --smooth-v / --no-smooth-v  Override value smoothing flag.\n");
    printf("  --pv-accum fp32|fp32+fp32|fp32+fp16  Override PV accumulation mode.\n");
    printf("  --compare flash|sage      Select reference for error metrics (default flash).\n");
    printf("  -h, --help          Show this help message.\n");
    printf("\nEnvironment overrides:\n");
    printf("  GGML_SAGE_SMOOTH_K=0|1, GGML_SAGE_SMOOTH_V=0|1, GGML_SAGE_PV_ACCUM=fp32|fp32+fp32|fp32+fp16\n");
}

static std::vector<std::string> split_case_list(const std::string & csv) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start < csv.size()) {
        size_t end = csv.find(',', start);
        if (end == std::string::npos) {
            end = csv.size();
        }
        if (end > start) {
            out.emplace_back(csv.substr(start, end - start));
        }
        start = end + 1;
    }
    return out;
}

static bool parse_cli(int argc, char ** argv, cli_options & opts) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--case") {
            if (i + 1 >= argc) {
                fprintf(stderr, "--case requires a name\n");
                return false;
            }
            opts.case_filters.emplace_back(argv[++i]);
        } else if (arg == "--cases") {
            if (i + 1 >= argc) {
                fprintf(stderr, "--cases requires a comma-separated list\n");
                return false;
            }
            auto names = split_case_list(argv[++i]);
            opts.case_filters.insert(opts.case_filters.end(), names.begin(), names.end());
        } else if (arg == "--iters") {
            if (i + 1 >= argc) {
                fprintf(stderr, "--iters requires a value\n");
                return false;
            }
            opts.iters = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--seed") {
            if (i + 1 >= argc) {
                fprintf(stderr, "--seed requires a value\n");
                return false;
            }
            opts.seed = strtoull(argv[++i], nullptr, 10);
        } else if (arg == "--list") {
            opts.list_only = true;
        } else if (arg == "--help" || arg == "-h") {
            opts.show_help = true;
            return true;
        } else if (arg == "--verbose") {
            opts.verbose = true;
        } else if (arg == "--load") {
            if (i + 1 >= argc) {
                fprintf(stderr, "--load requires a prefix\n");
                return false;
            }
            opts.load_prefix = argv[++i];
        } else if (arg == "--granularity") {
            if (i + 1 >= argc) {
                fprintf(stderr, "--granularity requires a value (per_warp or per_thread)\n");
                return false;
            }
            std::string val = argv[++i];
            if (val == "per_warp") {
                opts.granularity_override = GGML_SAGE_QK_GRANULARITY_PER_WARP;
            } else if (val == "per_thread") {
                opts.granularity_override = GGML_SAGE_QK_GRANULARITY_PER_THREAD;
            } else {
                fprintf(stderr, "unknown granularity '%s'\n", val.c_str());
                return false;
            }
            opts.granularity_override_set = true;
        } else if (arg == "--smooth-k") {
            opts.smooth_k_override_set = true;
            opts.smooth_k_override = true;
        } else if (arg == "--no-smooth-k") {
            opts.smooth_k_override_set = true;
            opts.smooth_k_override = false;
        } else if (arg == "--smooth-v") {
            opts.smooth_v_override_set = true;
            opts.smooth_v_override = true;
        } else if (arg == "--no-smooth-v") {
            opts.smooth_v_override_set = true;
            opts.smooth_v_override = false;
        } else if (arg == "--pv-accum") {
            if (i + 1 >= argc) {
                fprintf(stderr, "--pv-accum requires a value (fp32, fp32+fp32, fp32+fp16)\n");
                return false;
            }
            std::string val = argv[++i];
            if (!parse_pv_accum_string(val, opts.pv_accum_override)) {
                fprintf(stderr, "unknown pv-accum '%s'\n", val.c_str());
                return false;
            }
            opts.pv_accum_override_set = true;
        } else if (arg == "--compare") {
            if (i + 1 >= argc) {
                fprintf(stderr, "--compare requires a value (flash or sage)\n");
                return false;
            }
            std::string val = argv[++i];
            if (val == "flash") {
                opts.compare = cli_options::compare_mode::flash;
            } else if (val == "sage") {
                opts.compare = cli_options::compare_mode::sage;
            } else {
                fprintf(stderr, "unknown compare target '%s'\n", val.c_str());
                return false;
            }
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            return false;
        }
    }
    return true;
}

int main(int argc, char ** argv) {
#ifndef GGML_USE_CUDA
    printf("SageAttention test requires CUDA backend\n");
    return 0;
#else
    ggml_time_init();

    cli_options opts;
    if (!parse_cli(argc, argv, opts)) {
        print_usage(argv[0]);
        return 1;
    }
    if (opts.show_help) {
        print_usage(argv[0]);
        return 0;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        printf("failed to initialize CUDA backend\n");
        return 1;
    }

    std::vector<sage_case> cases = {
        {"b2_seq256_d128", 128, 256, 256, 16, 8, 2, false, true, GGML_SAGE_QK_GRANULARITY_PER_WARP, false, GGML_SAGE_PV_ACCUM_FP32_FP16},
        {"b1_causal_seq128_d64", 64, 128, 128, 12, 6, 1, true, true, GGML_SAGE_QK_GRANULARITY_PER_WARP, false, GGML_SAGE_PV_ACCUM_FP32_FP16},
        {"b1_causal_seq128_d64_nosmooth", 64, 128, 128, 12, 6, 1, true, false, GGML_SAGE_QK_GRANULARITY_PER_WARP, false, GGML_SAGE_PV_ACCUM_FP32_FP16},
        {"b4_seq512_d128", 128, 512, 512, 8, 4, 4, false, false, GGML_SAGE_QK_GRANULARITY_PER_WARP, false, GGML_SAGE_PV_ACCUM_FP32_FP16},
        {"b1_long_seq768_d128_causal", 128, 768, 768, 4, 4, 1, true, false, GGML_SAGE_QK_GRANULARITY_PER_WARP, false, GGML_SAGE_PV_ACCUM_FP32_FP16},
        {"b2_seq256_d128_thread", 128, 256, 256, 16, 8, 2, false, true, GGML_SAGE_QK_GRANULARITY_PER_THREAD, false, GGML_SAGE_PV_ACCUM_FP32_FP16},
        {"b1_causal_seq128_d64_thread", 64, 128, 128, 12, 6, 1, true, true, GGML_SAGE_QK_GRANULARITY_PER_THREAD, false, GGML_SAGE_PV_ACCUM_FP32_FP16},
        {"b1_causal_seq128_d64_smoothv", 64, 128, 128, 12, 6, 1, true, true, GGML_SAGE_QK_GRANULARITY_PER_WARP, true, GGML_SAGE_PV_ACCUM_FP32},
        {"b2_seq256_d128_fp32", 128, 256, 256, 16, 8, 2, false, true, GGML_SAGE_QK_GRANULARITY_PER_WARP, false, GGML_SAGE_PV_ACCUM_FP32},
        {"b2_seq256_d128_fp32fp32", 128, 256, 256, 16, 8, 2, false, true, GGML_SAGE_QK_GRANULARITY_PER_WARP, false, GGML_SAGE_PV_ACCUM_FP32_FP32},
    };

    external_data loaded;
    if (!opts.load_prefix.empty()) {
        if (!load_external_case(opts.load_prefix, loaded)) {
            fprintf(stderr, "failed to load case from %s\n", opts.load_prefix.c_str());
            ggml_backend_free(backend);
            return 1;
        }
        loaded.cfg.name = opts.load_prefix;
        cases = { loaded.cfg };
        opts.case_filters.clear();
    }

    if (opts.compare == cli_options::compare_mode::sage && !loaded.active) {
        fprintf(stderr, "--compare sage requires --load with sage reference tensors\n");
        ggml_backend_free(backend);
        return 1;
    }

    auto apply_overrides = [&](sage_case & cfg) {
        if (opts.granularity_override_set) {
            cfg.granularity = opts.granularity_override;
        }
        if (opts.smooth_k_override_set) {
            cfg.smooth_k = opts.smooth_k_override;
        }
        if (opts.smooth_v_override_set) {
            cfg.smooth_v = opts.smooth_v_override;
        }
        if (opts.pv_accum_override_set) {
            cfg.pv_accum = opts.pv_accum_override;
        }
    };

    if (loaded.active) {
        apply_overrides(loaded.cfg);
    }
    for (auto & cfg : cases) {
        apply_overrides(cfg);
    }

    if (opts.list_only) {
        printf("Available cases:\n");
        for (const auto & cfg : cases) {
            printf("  %s\n", cfg.name.c_str());
        }
        ggml_backend_free(backend);
        return 0;
    }

    if (opts.iters <= 0) {
        fprintf(stderr, "Iteration count must be positive\n");
        ggml_backend_free(backend);
        return 1;
    }

    struct case_entry {
        const sage_case * cfg;
        size_t index;
    };
    std::vector<case_entry> selected;

    if (opts.case_filters.empty()) {
        for (size_t i = 0; i < cases.size(); ++i) {
            selected.push_back({&cases[i], i});
        }
    } else {
        for (const std::string & name : opts.case_filters) {
            bool found = false;
            for (size_t i = 0; i < cases.size(); ++i) {
                if (name == cases[i].name) {
                    selected.push_back({&cases[i], i});
                    found = true;
                    break;
                }
            }
            if (!found) {
                fprintf(stderr, "Unknown case '%s'\n", name.c_str());
                ggml_backend_free(backend);
                return 1;
            }
        }
    }

    double best_speedup = -1.0;
    double worst_rms = -1.0;
    std::string best_case = "n/a";
    std::string worst_case = "n/a";

    for (const case_entry & entry : selected) {
        case_result res;
        std::vector<float> exported_sage;
        std::vector<float> exported_flash;
        if (loaded.active) {
            if (opts.iters != 1 && entry.index == 0) {
                fprintf(stderr, "warning: overriding iters to 1 for loaded case\n");
                opts.iters = 1;
            }
            res = run_case(*entry.cfg, backend, opts.seed, &loaded, &exported_sage, &exported_flash, opts.compare);
        } else {
            const uint64_t seed = opts.seed + (uint64_t) entry.index * 1000003ull;
            res = run_case_iters(*entry.cfg, backend, seed, opts.iters);
        }
        print_case_report(*entry.cfg, res, opts.iters, opts.verbose);

        if (loaded.active && entry.index == 0) {
            auto compare_to_ref = [](const char * label, const std::vector<float> & ref, const std::vector<float> & cur) {
                if (ref.empty() || ref.size() != cur.size()) {
                    return;
                }
                double max_diff = 0.0;
                double sum_sq = 0.0;
                size_t max_idx = 0;
                for (size_t i = 0; i < ref.size(); ++i) {
                    const double diff = (double) cur[i] - (double) ref[i];
                    const double abs_diff = std::abs(diff);
                    if (abs_diff > max_diff) {
                        max_diff = abs_diff;
                        max_idx = i;
                    }
                    sum_sq += diff*diff;
                }
                const double rms = std::sqrt(sum_sq / ref.size());
                printf("  %s ref diff: max %.3e rms %.3e (idx %zu)\n", label, max_diff, rms, max_idx);
            };
            compare_to_ref("sage_vs_ref", loaded.ref_sage, exported_sage);
            compare_to_ref("flash_vs_ref", loaded.ref_flash, exported_flash);
        }

        const double speedup = (res.sage_ms > 0.0) ? (res.flash_ms / res.sage_ms) : 0.0;
        if (speedup > best_speedup) {
            best_speedup = speedup;
            best_case = entry.cfg->name;
        }
        if (res.rms_diff > worst_rms) {
            worst_rms = res.rms_diff;
            worst_case = entry.cfg->name;
        }
    }

    printf("\nSummary\n");
    printf("  best speedup: %.2fx (%s)\n", best_speedup, best_case.c_str());
    printf("  worst rms diff: %.3e (%s)\n", worst_rms, worst_case.c_str());

    ggml_backend_free(backend);
    return 0;
#endif
}
