#define(VARIANTS)

[
  {
    "REPLS": {
      "TYPE": "f32"
    }
  },
  {
    "REPLS": {
      "TYPE": "f16"
    }
  }
]

#end(VARIANTS)

#define(SHADER)
enable f16;

@group(0) @binding(0)
var<storage, read> knl: array<{{TYPE}}>;

@group(0) @binding(1)
var<storage, read> src: array<f32>;

@group(0) @binding(2)
var<storage, read_write> dst: array<f32>;

struct Params {
    offset_knl: u32, // in elements
    offset_src: u32, // in elements
    offset_dst: u32, // in elements

    // Tensor shapes
    Cout: u32,
    Cin: u32,
    N: u32,
    KW: u32,
    KH: u32,
    W: u32,
    H: u32,
    OW: u32,
    OH: u32,

    // Conv params
    s0: u32,
    s1: u32,
    p0: u32,
    p1: u32,
    d0: u32,
    d1: u32,

    // Kernel strides (in elements)
    nb00: u32,
    nb01: u32,
    nb02: u32,
    nb03: u32,

    // Input strides (in elements)
    nb10: u32,
    nb11: u32,
    nb12: u32,
    nb13: u32,

    // Output strides (in elements)
    nb0: u32,
    nb1: u32,
    nb2: u32,
    nb3: u32,
};

@group(0) @binding(3)
var<uniform> params: Params;

const VEC_SIZE: u32 = 4u;

const BS_K: u32 = 64u;
const BS_NPQ: u32 = 128u;
const BS_CRS: u32 = 16u;

const TS_K: u32 = 4u;
const TS_NPQ: u32 = 8u;

const WG_K: u32 = BS_K / TS_K;       // 16
const WG_NPQ: u32 = BS_NPQ / TS_NPQ; // 8
const WG_SIZE: u32 = WG_K * WG_NPQ;  // 128

const BS_NPQ_VEC: u32 = BS_NPQ / VEC_SIZE; // 16
const TS_NPQ_VEC: u32 = TS_NPQ / VEC_SIZE; // 2

var<workgroup> Ash: array<{{TYPE}}, BS_CRS * BS_K>;
var<workgroup> Bsh: array<vec4<f32>, BS_CRS * BS_NPQ_VEC>;

fn split_work(work_size: u32, block_size: u32) -> u32 {
    return (work_size + block_size - 1u) / block_size;
}

// wid.x = K tile index, wid.y = NPQ tile index
@compute @workgroup_size(WG_K, WG_NPQ, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let K: u32 = params.Cout;
    let CRS: u32 = params.Cin * params.KH * params.KW;
    let NPQ: u32 = params.N * params.OH * params.OW;
    let OHOW: u32 = params.OH * params.OW;
    let KWH: u32 = params.KW * params.KH;
    let unit_stride_dilation: bool =
        params.s0 == 1u && params.s1 == 1u && params.d0 == 1u && params.d1 == 1u;

    let tid: u32 = lid.y * WG_K + lid.x;

    let offset_k: u32 = wid.x * BS_K;
    let offset_npq: u32 = wid.y * BS_NPQ;

    var regC: array<array<vec4<f32>, TS_NPQ_VEC>, TS_K>;
    for (var i: u32 = 0u; i < TS_K; i = i + 1u) {
        for (var j: u32 = 0u; j < TS_NPQ_VEC; j = j + 1u) {
            regC[i][j] = vec4<f32>(0.0);
        }
    }

    // Precompute: kernel contiguity check (eliminates CRS decomposition in A-tile loading)
    let contiguous_knl: bool = (params.nb00 == 1u && params.nb01 == params.KW && params.nb02 == KWH);

    // Precompute: f32 reciprocals for fast integer division (~4 cycles vs ~20-30 for integer div)
    // Safe for all values < 2^24 (our max CRS=2880, NPQ~1M, well within range)
    let inv_KWH: f32 = 1.0 / f32(KWH);
    let inv_KW: f32 = 1.0 / f32(params.KW);
    let inv_OHOW: f32 = 1.0 / f32(OHOW);
    let inv_OW: f32 = 1.0 / f32(params.OW);

    // Precompute: NPQ decomposition for this thread's B-tile column
    // Each thread always loads the same npq_l_vec across all iterations (since WG_SIZE % BS_NPQ_VEC == 0)
    let my_npq_l_vec: u32 = tid % BS_NPQ_VEC;
    let my_npq_g_base: u32 = offset_npq + my_npq_l_vec * 4u;
    let my_full_vec_in_range: bool = (my_npq_g_base + 3u) < NPQ;
    var my_n: u32 = 0u;
    var my_oh: u32 = 0u;
    var my_ow: u32 = 0u;
    var my_same_row: bool = false;
    if (my_full_vec_in_range) {
        my_n = u32(f32(my_npq_g_base) * inv_OHOW);
        let my_pq: u32 = my_npq_g_base - my_n * OHOW;
        my_oh = u32(f32(my_pq) * inv_OW);
        my_ow = my_pq - my_oh * params.OW;
        my_same_row = (my_ow + 4u) <= params.OW;
    }

    let nb_crs: u32 = split_work(CRS, BS_CRS);
    for (var b_idx_crs: u32 = 0u; b_idx_crs < nb_crs; b_idx_crs = b_idx_crs + 1u) {
        let offset_crs: u32 = b_idx_crs * BS_CRS;

        // ---- Load A-tile (kernel weights) into shared memory [CRS][K] layout ----
        var i0: u32 = tid;
        while (i0 < BS_K * BS_CRS) {
            let k_l: u32 = i0 / BS_CRS;
            let crs_l: u32 = i0 % BS_CRS;
            let k_g: u32 = offset_k + k_l;
            let crs_g: u32 = offset_crs + crs_l;

            var w_val: {{TYPE}} = {{TYPE}}(0.0);
            if (k_g < K && crs_g < CRS) {
                if (contiguous_knl) {
                    w_val = knl[params.offset_knl + k_g * params.nb03 + crs_g];
                } else {
                    let cin_idx: u32 = u32(f32(crs_g) * inv_KWH);
                    let crs_rem: u32 = crs_g - cin_idx * KWH;
                    let kh_idx: u32 = u32(f32(crs_rem) * inv_KW);
                    let kw_idx: u32 = crs_rem - kh_idx * params.KW;
                    let knl_idx: u32 =
                        params.offset_knl +
                        kw_idx * params.nb00 +
                        kh_idx * params.nb01 +
                        cin_idx * params.nb02 +
                        k_g * params.nb03;
                    w_val = knl[knl_idx];
                }
            }
            Ash[crs_l * BS_K + k_l] = w_val;

            i0 = i0 + WG_SIZE;
        }

        // ---- Load B-tile (input data) ----
        var i1: u32 = tid;
        while (i1 < BS_CRS * BS_NPQ_VEC) {
            let crs_l: u32 = i1 / BS_NPQ_VEC;
            let npq_l_vec: u32 = i1 % BS_NPQ_VEC;
            let crs_g: u32 = offset_crs + crs_l;

            var vals: array<f32, 4>;
            vals[0] = 0.0;
            vals[1] = 0.0;
            vals[2] = 0.0;
            vals[3] = 0.0;

            if (crs_g < CRS) {
                let cin_idx: u32 = u32(f32(crs_g) * inv_KWH);
                let crs_rem: u32 = crs_g - cin_idx * KWH;
                let kh_idx: u32 = u32(f32(crs_rem) * inv_KW);
                let kw_idx: u32 = crs_rem - kh_idx * params.KW;

                // Use precomputed NPQ decomposition (npq_l_vec == my_npq_l_vec always)
                if (my_full_vec_in_range) {
                    let n_idx: u32 = my_n;
                    let oh_idx: u32 = my_oh;
                    let ow_idx: u32 = my_ow;
                    var loaded_fast: bool = false;

                    if (my_same_row) {
                        var h_idx: i32 = 0;
                        var w0: i32    = 0;
                        var w3: i32    = 0;
                        if (unit_stride_dilation) {
                            h_idx = i32(oh_idx + kh_idx) - i32(params.p1);
                            w0    = i32(ow_idx + kw_idx) - i32(params.p0);
                            w3    = w0 + 3;
                        } else {
                            h_idx = i32(oh_idx * params.s1 + kh_idx * params.d1) - i32(params.p1);
                            w0    = i32(ow_idx * params.s0 + kw_idx * params.d0) - i32(params.p0);
                            w3    = w0 + i32(3u * params.s0);
                        }

                        if (h_idx >= 0 && h_idx < i32(params.H) && w0 >= 0 && w3 < i32(params.W)) {
                            let src_base: u32 =
                                params.offset_src +
                                u32(w0) * params.nb10 +
                                u32(h_idx) * params.nb11 +
                                cin_idx * params.nb12 +
                                n_idx * params.nb13;
                            if (unit_stride_dilation) {
                                vals[0] = src[src_base + 0u * params.nb10];
                                vals[1] = src[src_base + 1u * params.nb10];
                                vals[2] = src[src_base + 2u * params.nb10];
                                vals[3] = src[src_base + 3u * params.nb10];
                            } else {
                                vals[0] = src[src_base + 0u * params.s0 * params.nb10];
                                vals[1] = src[src_base + 1u * params.s0 * params.nb10];
                                vals[2] = src[src_base + 2u * params.s0 * params.nb10];
                                vals[3] = src[src_base + 3u * params.s0 * params.nb10];
                            }
                            loaded_fast = true;
                        }
                    } else {
                        loaded_fast = false;
                    }

                    if (!loaded_fast) {
                        for (var v: u32 = 0u; v < 4u; v = v + 1u) {
                            let npq_g: u32 = my_npq_g_base + v;
                            let n_v: u32   = u32(f32(npq_g) * inv_OHOW);
                            let pq_v: u32  = npq_g - n_v * OHOW;
                            let oh_v: u32  = u32(f32(pq_v) * inv_OW);
                            let ow_v: u32  = pq_v - oh_v * params.OW;

                            var h_idx: i32 = 0;
                            var w_idx: i32 = 0;
                            if (unit_stride_dilation) {
                                h_idx = i32(oh_v + kh_idx) - i32(params.p1);
                                w_idx = i32(ow_v + kw_idx) - i32(params.p0);
                            } else {
                                h_idx = i32(oh_v * params.s1 + kh_idx * params.d1) - i32(params.p1);
                                w_idx = i32(ow_v * params.s0 + kw_idx * params.d0) - i32(params.p0);
                            }
                            if (h_idx >= 0 && h_idx < i32(params.H) && w_idx >= 0 && w_idx < i32(params.W)) {
                                let src_idx: u32 =
                                    params.offset_src +
                                    u32(w_idx) * params.nb10 +
                                    u32(h_idx) * params.nb11 +
                                    cin_idx * params.nb12 +
                                    n_v * params.nb13;
                                vals[v] = src[src_idx];
                            }
                        }
                    }
                } else {
                    for (var v: u32 = 0u; v < 4u; v = v + 1u) {
                        let npq_g: u32 = my_npq_g_base + v;
                        if (npq_g < NPQ) {
                            let n_idx: u32 = u32(f32(npq_g) * inv_OHOW);
                            let pq_idx: u32 = npq_g - n_idx * OHOW;
                            let oh_idx: u32 = u32(f32(pq_idx) * inv_OW);
                            let ow_idx: u32 = pq_idx - oh_idx * params.OW;

                            var h_idx: i32 = 0;
                            var w_idx: i32 = 0;
                            if (unit_stride_dilation) {
                                h_idx = i32(oh_idx + kh_idx) - i32(params.p1);
                                w_idx = i32(ow_idx + kw_idx) - i32(params.p0);
                            } else {
                                h_idx = i32(oh_idx * params.s1 + kh_idx * params.d1) - i32(params.p1);
                                w_idx = i32(ow_idx * params.s0 + kw_idx * params.d0) - i32(params.p0);
                            }
                            if (h_idx >= 0 && h_idx < i32(params.H) && w_idx >= 0 && w_idx < i32(params.W)) {
                                let src_idx: u32 =
                                    params.offset_src +
                                    u32(w_idx) * params.nb10 +
                                    u32(h_idx) * params.nb11 +
                                    cin_idx * params.nb12 +
                                    n_idx * params.nb13;
                                vals[v] = src[src_idx];
                            }
                        }
                    }
                }
            }

            Bsh[crs_l * BS_NPQ_VEC + npq_l_vec] = vec4<f32>(vals[0], vals[1], vals[2], vals[3]);

            i1 = i1 + WG_SIZE;
        }

        workgroupBarrier();

        // ---- Compute ----
        for (var crs_l: u32 = 0u; crs_l < BS_CRS; crs_l = crs_l + 1u) {
            var regA: array<f32, TS_K>;
            for (var k_l_reg: u32 = 0u; k_l_reg < TS_K; k_l_reg = k_l_reg + 1u) {
                regA[k_l_reg] = f32(Ash[crs_l * BS_K + lid.x * TS_K + k_l_reg]);
            }

            for (var npq_l_vec_reg: u32 = 0u; npq_l_vec_reg < TS_NPQ_VEC; npq_l_vec_reg = npq_l_vec_reg + 1u) {
                let regB: vec4<f32> = Bsh[crs_l * BS_NPQ_VEC + lid.y * TS_NPQ_VEC + npq_l_vec_reg];

                for (var k_l_reg: u32 = 0u; k_l_reg < TS_K; k_l_reg = k_l_reg + 1u) {
                    regC[k_l_reg][npq_l_vec_reg] =
                        regC[k_l_reg][npq_l_vec_reg] + vec4<f32>(regA[k_l_reg]) * regB;
                }
            }
        }

        workgroupBarrier();
    }

    // ---- Write results ----
    for (var k_l_reg: u32 = 0u; k_l_reg < TS_K; k_l_reg = k_l_reg + 1u) {
        let k_g: u32 = offset_k + lid.x * TS_K + k_l_reg;
        if (k_g >= K) {
            continue;
        }

        for (var npq_l_vec_reg: u32 = 0u; npq_l_vec_reg < TS_NPQ_VEC; npq_l_vec_reg = npq_l_vec_reg + 1u) {
            let npq_g_base: u32 = offset_npq + (lid.y * TS_NPQ_VEC + npq_l_vec_reg) * 4u;
            if (npq_g_base >= NPQ) {
                continue;
            }

            let res: vec4<f32> = regC[k_l_reg][npq_l_vec_reg];

            let n_idx: u32 = u32(f32(npq_g_base) * inv_OHOW);
            let pq_idx: u32 = npq_g_base - n_idx * OHOW;
            let oh_idx: u32 = u32(f32(pq_idx) * inv_OW);
            let ow_idx: u32 = pq_idx - oh_idx * params.OW;

            let contiguous_row: bool =
                params.nb0 == 1u &&
                params.nb1 == params.OW &&
                (ow_idx + 4u) <= params.OW &&
                (npq_g_base + 4u) <= NPQ;

            if (contiguous_row) {
                let dst_idx: u32 =
                    params.offset_dst +
                    ow_idx * params.nb0 +
                    oh_idx * params.nb1 +
                    k_g * params.nb2 +
                    n_idx * params.nb3;
                dst[dst_idx + 0u] = res.x;
                dst[dst_idx + 1u] = res.y;
                dst[dst_idx + 2u] = res.z;
                dst[dst_idx + 3u] = res.w;
            } else {
                let vals: array<f32, 4> = array<f32, 4>(res.x, res.y, res.z, res.w);
                for (var v: u32 = 0u; v < 4u; v = v + 1u) {
                    let npq_g: u32 = npq_g_base + v;
                    if (npq_g < NPQ) {
                        let n_s: u32 = u32(f32(npq_g) * inv_OHOW);
                        let pq_s: u32 = npq_g - n_s * OHOW;
                        let oh_s: u32 = u32(f32(pq_s) * inv_OW);
                        let ow_s: u32 = pq_s - oh_s * params.OW;
                        let dst_idx_s: u32 =
                            params.offset_dst +
                            ow_s * params.nb0 +
                            oh_s * params.nb1 +
                            k_g * params.nb2 +
                            n_s * params.nb3;
                        dst[dst_idx_s] = vals[v];
                    }
                }
            }
        }
    }
}

#end(SHADER)
