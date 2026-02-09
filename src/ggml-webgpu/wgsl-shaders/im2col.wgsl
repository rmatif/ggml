#define(VARIANTS)

[
  {
    "SHADER_SUFFIX": "f32",
    "REPLS": {
      "DST_TYPE": "f32"
    }
  },
  {
    "SHADER_SUFFIX": "f16",
    "DECLS": ["F16"],
    "REPLS": {
      "DST_TYPE": "f16"
    }
  }
]

#end(VARIANTS)

#define(DECLS)

#decl(F16)
enable f16;
#enddecl(F16)

#end(DECLS)

#define(SHADER)

DECLS

@group(0) @binding(0)
var<storage, read_write> src: array<f32>;

@group(0) @binding(1)
var<storage, read_write> dst: array<{{DST_TYPE}}>;

struct Params {
    offset_src: u32, // in elements
    offset_dst: u32, // in elements

    // Source strides in elements for batch/channel addressing
    batch_offset: u32,
    offset_delta: u32,

    // Tensor and kernel shape
    IC: u32,
    IW: u32,
    IH: u32,
    OW: u32,
    OH: u32,
    KW: u32,
    KH: u32,

    // Flattened per-(batch, channel) workload
    pelements: u32,
    CHW: u32,

    // Convolution parameters
    s0: i32,
    s1: i32,
    p0: i32,
    p1: i32,
    d0: i32,
    d1: i32,

    batch_ic: u32,

    // Destination strides in elements
    stride_dst0: u32,
    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,
    is_2D: u32,

    // Fast division precomputed multipliers
    pelements_mp: u32,
    IC_mp: u32,
    ksize_mp: u32,
    OW_mp: u32,

    // Vectorized path configuration
    pelements_vec: u32,
    pelements_vec_mp: u32,
    im2col_fast_path: u32,
};

@group(0) @binding(2)
var<uniform> params: Params;

override wg_size: u32;

fn mulhi_u32(a: u32, b: u32) -> u32 {
    let a_lo: u32 = a & 0xffffu;
    let a_hi: u32 = a >> 16u;
    let b_lo: u32 = b & 0xffffu;
    let b_hi: u32 = b >> 16u;

    let p0: u32 = a_lo * b_lo;
    let p1: u32 = a_lo * b_hi;
    let p2: u32 = a_hi * b_lo;
    let p3: u32 = a_hi * b_hi;

    let carry: u32 = (p0 >> 16u) + (p1 & 0xffffu) + (p2 & 0xffffu);
    return p3 + (p1 >> 16u) + (p2 >> 16u) + (carry >> 16u);
}

fn fastdiv_u32(n: u32, mp: u32, L: u32) -> u32 {
    return (mulhi_u32(n, mp) + n) >> L;
}

fn decode_im2col_i(i: u32, ksize: u32, ksize_mp: u32, ksize_L: u32, ow: u32, ow_mp: u32, ow_L: u32) -> vec3<u32> {
    let kx: u32 = fastdiv_u32(i, ksize_mp, ksize_L);
    let rem: u32 = i - kx * ksize;
    let ky: u32 = fastdiv_u32(rem, ow_mp, ow_L);
    let ix: u32 = rem - ky * ow;
    return vec3<u32>(kx, ky, ix);
}

@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.y >= params.OH || params.pelements == 0u) {
        return;
    }

    let oh = gid.y;
    let ksize = params.OW * params.KH;
    let pelements_L = 32u - countLeadingZeros(params.pelements - 1u);
    let ic_L = 32u - countLeadingZeros(params.IC - 1u);
    let ksize_L = 32u - countLeadingZeros(ksize - 1u);
    let ow_L = 32u - countLeadingZeros(params.OW - 1u);

    // Common SDXL UNet case: stride=1, dilation=1. Process 4 adjacent ix values per invocation.
    if (params.im2col_fast_path != 0u) {
        let flat_vec_idx = gid.x;
        let pelements_vec_L = 32u - countLeadingZeros(params.pelements_vec - 1u);
        let batch_ic = fastdiv_u32(flat_vec_idx, params.pelements_vec_mp, pelements_vec_L);
        if (batch_ic >= params.batch_ic) {
            return;
        }

        let vec_i = flat_vec_idx - batch_ic * params.pelements_vec;
        let i_base = vec_i * 4u;
        if (i_base >= params.pelements) {
            return;
        }

        let batch = fastdiv_u32(batch_ic, params.IC_mp, ic_L);
        let ic = batch_ic - batch * params.IC;

        let i0_dec = decode_im2col_i(i_base, ksize, params.ksize_mp, ksize_L, params.OW, params.OW_mp, ow_L);
        let kx = i0_dec.x;
        let ky = i0_dec.y;
        let ix = i0_dec.z;

        let src_base = params.offset_src + ic * params.offset_delta + batch * params.batch_offset;

        var dst_batch_base = params.offset_dst;
        if (params.is_2D != 0u) {
            dst_batch_base += batch * params.stride_dst3 + oh * params.stride_dst2;
        } else {
            // 1D IM2COL output layout is [CHW, OW, N, 1], so batch is dim2.
            dst_batch_base += batch * params.stride_dst2;
        }

        // Fast row path: 4 adjacent ix in the same kernel position.
        if (ix + 3u < params.OW) {
            var vals: array<f32, 4>;
            vals[0] = 0.0;
            vals[1] = 0.0;
            vals[2] = 0.0;
            vals[3] = 0.0;

            let iih = i32(oh) * params.s1 + i32(ky) * params.d1 - params.p1;
            let iiw0 = i32(ix) * params.s0 + i32(kx) * params.d0 - params.p0;
            if (iih >= 0 && iiw0 >= 0 && u32(iih) < params.IH && u32(iiw0 + 3) < params.IW) {
                let src_row = src_base + u32(iih) * params.IW + u32(iiw0);
                vals[0] = src[src_row + 0u];
                vals[1] = src[src_row + 1u];
                vals[2] = src[src_row + 2u];
                vals[3] = src[src_row + 3u];
            } else {
                for (var v: u32 = 0u; v < 4u; v = v + 1u) {
                    let ix_v = ix + v;
                    let iiw = i32(ix_v) * params.s0 + i32(kx) * params.d0 - params.p0;
                    if (iih >= 0 && iiw >= 0 && u32(iih) < params.IH && u32(iiw) < params.IW) {
                        vals[v] = src[src_base + u32(iih) * params.IW + u32(iiw)];
                    }
                }
            }

            let dst_ch = ic * (params.KW * params.KH) + ky * params.KW + kx;
            let dst_ch_base = dst_batch_base + dst_ch * params.stride_dst0;
            for (var v: u32 = 0u; v < 4u; v = v + 1u) {
                let i_v = i_base + v;
                if (i_v < params.pelements) {
                    let dst_idx = dst_ch_base + (ix + v) * params.stride_dst1;
                    dst[dst_idx] = {{DST_TYPE}}(vals[v]);
                }
            }
            return;
        }

        // Boundary fallback for vectorized mode (crosses OW boundary or tail).
        for (var v: u32 = 0u; v < 4u; v = v + 1u) {
            let i_v = i_base + v;
            if (i_v >= params.pelements) {
                continue;
            }

            let iv_dec = decode_im2col_i(i_v, ksize, params.ksize_mp, ksize_L, params.OW, params.OW_mp, ow_L);
            let kx_v = iv_dec.x;
            let ky_v = iv_dec.y;
            let ix_v = iv_dec.z;

            let iiw = i32(ix_v) * params.s0 + i32(kx_v) * params.d0 - params.p0;
            let iih = i32(oh) * params.s1 + i32(ky_v) * params.d1 - params.p1;

            var value: f32 = 0.0;
            if (iih >= 0 && iiw >= 0 && u32(iih) < params.IH && u32(iiw) < params.IW) {
                value = src[src_base + u32(iih) * params.IW + u32(iiw)];
            }

            let dst_ch = ic * (params.KW * params.KH) + ky_v * params.KW + kx_v;
            let dst_idx = dst_batch_base + ix_v * params.stride_dst1 + dst_ch * params.stride_dst0;
            dst[dst_idx] = {{DST_TYPE}}(value);
        }
        return;
    }

    // Generic scalar path.
    let flat_idx = gid.x;
    let batch_ic = fastdiv_u32(flat_idx, params.pelements_mp, pelements_L);
    if (batch_ic >= params.batch_ic) {
        return;
    }

    let i = flat_idx - batch_ic * params.pelements;
    let batch = fastdiv_u32(batch_ic, params.IC_mp, ic_L);
    let ic = batch_ic - batch * params.IC;
    let i_dec = decode_im2col_i(i, ksize, params.ksize_mp, ksize_L, params.OW, params.OW_mp, ow_L);
    let kx = i_dec.x;
    let ky = i_dec.y;
    let ix = i_dec.z;

    let iiw = i32(ix) * params.s0 + i32(kx) * params.d0 - params.p0;
    let iih = i32(oh) * params.s1 + i32(ky) * params.d1 - params.p1;

    var value: f32 = 0.0;
    if (iih >= 0 && iiw >= 0 && u32(iih) < params.IH && u32(iiw) < params.IW) {
        let src_base = params.offset_src + ic * params.offset_delta + batch * params.batch_offset;
        value = src[src_base + u32(iih) * params.IW + u32(iiw)];
    }

    let dst_ch = ic * (params.KW * params.KH) + ky * params.KW + kx;
    var dst_idx = params.offset_dst + ix * params.stride_dst1 + dst_ch * params.stride_dst0;
    if (params.is_2D != 0u) {
        dst_idx += batch * params.stride_dst3 + oh * params.stride_dst2;
    } else {
        // 1D IM2COL output layout is [CHW, OW, N, 1], so batch is dim2.
        dst_idx += batch * params.stride_dst2;
    }
    dst[dst_idx] = {{DST_TYPE}}(value);
}

#end(SHADER)
