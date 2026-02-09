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
};

@group(0) @binding(2)
var<uniform> params: Params;

override wg_size: u32;

@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.y >= params.OH || params.pelements == 0u) {
        return;
    }

    let flat_idx = gid.x;
    let batch_ic = flat_idx / params.pelements;
    if (batch_ic >= params.batch_ic) {
        return;
    }

    let i = flat_idx - batch_ic * params.pelements;
    let oh = gid.y;
    let batch = batch_ic / params.IC;
    let ic = batch_ic - batch * params.IC;

    let ksize = params.OW * params.KH;
    let kx = i / ksize;
    let rem = i - kx * ksize;
    let ky = rem / params.OW;
    let ix = rem - ky * params.OW;

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
