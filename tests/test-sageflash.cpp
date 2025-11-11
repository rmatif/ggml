#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <random>
#include <string>
#include <vector>

struct sage_case {
    const char * name;
    int head_dim;
    int seq_q;
    int seq_k;
    int num_q_heads;
    int num_k_heads;
    int batch;
    bool is_causal;
    bool smooth_k;
    ggml_sage_qk_granularity granularity;
};

struct case_result {
    double sage_ms;
    double flash_ms;
    double max_diff;
    double rms_diff;
};

struct cli_options {
    std::vector<std::string> case_filters;
    int iters = 1;
    uint64_t seed = 1234;
    bool list_only = false;
    bool show_help = false;
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

static case_result run_case(const sage_case & cfg, ggml_backend_t backend, uint64_t seed) {
    std::mt19937 rng(seed);

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

    ggml_tensor * sage = ggml_sage_attn_sm89_fp16(ctx, q, k, v, scale, cfg.is_causal, cfg.smooth_k, cfg.granularity);
    ggml_tensor * sage_f32 = ggml_cast(ctx, sage, GGML_TYPE_F32);

    ggml_tensor * qf = ggml_cast(ctx, q, GGML_TYPE_F32);
    ggml_tensor * kf = ggml_cast(ctx, k, GGML_TYPE_F32);
    ggml_tensor * vf = ggml_cast(ctx, v, GGML_TYPE_F32);
    ggml_tensor * flash = ggml_flash_attn_ext(ctx, qf, kf, vf, nullptr, scale, 0.0f, 0.0f);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer);

    fill_tensor_uniform(q, rng);
    fill_tensor_uniform(k, rng);
    fill_tensor_uniform(v, rng);

    std::vector<float> sage_out;
    std::vector<float> flash_out;

    case_result res = {};
    res.sage_ms = run_graph(ctx, backend, sage_f32, sage_out);
    res.flash_ms = run_graph(ctx, backend, flash, flash_out);

    auto has_non_finite = [](const std::vector<float> & vals) {
        for (float v : vals) {
            if (!std::isfinite(v)) {
                return true;
            }
        }
        return false;
    };

    if (has_non_finite(sage_out) || has_non_finite(flash_out)) {
        const bool sage_bad = has_non_finite(sage_out);
        const bool flash_bad = has_non_finite(flash_out);
        printf("WARNING[%s]: non-finite outputs detected (sage=%d, flash=%d)\n",
               cfg.name,
               sage_bad ? 1 : 0,
               flash_bad ? 1 : 0);
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
        res.max_diff = 0.0;
        double sum_sq = 0.0;
        for (size_t i = 0; i < sage_out.size(); ++i) {
            const double diff = (double) sage_out[i] - (double) flash_out[i];
            res.max_diff = std::max(res.max_diff, std::abs(diff));
            sum_sq += diff*diff;
        }
        res.rms_diff = std::sqrt(sum_sq / sage_out.size());
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);

    return res;
}

static case_result run_case_iters(const sage_case & cfg, ggml_backend_t backend, uint64_t seed_base, int iters) {
    GGML_ASSERT(iters > 0);
    case_result agg = {};
    double sage_sum = 0.0;
    double flash_sum = 0.0;
    double rms_sum = 0.0;
    double max_diff = 0.0;

    for (int it = 0; it < iters; ++it) {
        const uint64_t seed = seed_base + (uint64_t) it * 10007ull;
        const case_result cur = run_case(cfg, backend, seed);
        sage_sum += cur.sage_ms;
        flash_sum += cur.flash_ms;
        rms_sum += cur.rms_diff;
        max_diff = std::max(max_diff, cur.max_diff);
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

static void print_case_report(const sage_case & cfg, const case_result & res, int iters) {
    const double speedup = (res.sage_ms > 0.0) ? (res.flash_ms / res.sage_ms) : 0.0;
    printf("\nCase %s\n", cfg.name);
    printf("  shape: batch=%d seq_q=%d seq_k=%d nq=%d nk=%d d=%d causal=%d smooth_k=%d\n",
           cfg.batch, cfg.seq_q, cfg.seq_k, cfg.num_q_heads, cfg.num_k_heads, cfg.head_dim,
           cfg.is_causal ? 1 : 0, cfg.smooth_k ? 1 : 0);
    printf("  granularity: %s\n", granularity_name(cfg.granularity));
    printf("  iterations: %d\n", iters);
    printf("  timings (avg): sage %.3f ms, flash %.3f ms (speedup %.2fx)\n", res.sage_ms, res.flash_ms, speedup);
    printf("  diff: max %.3e rms_avg %.3e\n", res.max_diff, res.rms_diff);
}

static void print_usage(const char * prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --case <name>       Run only the specified case (can be repeated).\n");
    printf("  --cases a,b,c       Comma-separated list of case names to run.\n");
    printf("  --iters <n>         Number of iterations per case (default 1).\n");
    printf("  --seed <value>      Base RNG seed (default 1234).\n");
    printf("  --list              List available case names and exit.\n");
    printf("  -h, --help          Show this help message.\n");
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

    const std::vector<sage_case> cases = {
        {"b2_seq256_d128", 128, 256, 256, 16, 8, 2, false, true, GGML_SAGE_QK_GRANULARITY_PER_WARP},
        {"b1_causal_seq128_d64", 64, 128, 128, 12, 6, 1, true, true, GGML_SAGE_QK_GRANULARITY_PER_WARP},
        {"b4_seq512_d128", 128, 512, 512, 8, 4, 4, false, false, GGML_SAGE_QK_GRANULARITY_PER_WARP},
        {"b1_long_seq768_d128_causal", 128, 768, 768, 4, 4, 1, true, false, GGML_SAGE_QK_GRANULARITY_PER_WARP},
        {"b2_seq256_d128_thread", 128, 256, 256, 16, 8, 2, false, true, GGML_SAGE_QK_GRANULARITY_PER_THREAD},
    };

    if (opts.list_only) {
        printf("Available cases:\n");
        for (const auto & cfg : cases) {
            printf("  %s\n", cfg.name);
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
        const uint64_t seed = opts.seed + (uint64_t) entry.index * 1000003ull;
        const case_result res = run_case_iters(*entry.cfg, backend, seed, opts.iters);
        print_case_report(*entry.cfg, res, opts.iters);

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
