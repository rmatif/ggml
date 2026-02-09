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
  },
  {
    "REPLS": {
      "TYPE": "i32"
    }
  }
]

#end(VARIANTS)

#define(SHADER)
enable f16;

@group(0) @binding(0)
var<storage, read_write> src: array<{{TYPE}}>;

@group(0) @binding(1)
var<storage, read_write> dst: array<{{TYPE}}>;

struct Params {
    ne: u32,
    offset_src: u32, // in elements
    offset_dst: u32, // in elements

    // Strides (in elements)
    stride_src0: u32,
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,

    stride_dst0: u32,
    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,

    // Shapes
    src_ne0: u32,
    src_ne1: u32,
    src_ne2: u32,
    src_ne3: u32,

    dst_ne0: u32,
    dst_ne1: u32,
    dst_ne2: u32,
    dst_ne3: u32,
};

@group(0) @binding(2)
var<uniform> params: Params;

override wg_size: u32;
@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.ne) {
        return;
    }

    let plane0 = params.dst_ne0;
    let plane1 = params.dst_ne0 * params.dst_ne1;
    let plane2 = plane1 * params.dst_ne2;

    var i = idx;
    let i3 = i / plane2;
    i = i % plane2;
    let i2 = i / plane1;
    i = i % plane1;
    let i1 = i / plane0;
    let i0 = i % plane0;

    let s0 = i0 % params.src_ne0;
    let s1 = i1 % params.src_ne1;
    let s2 = i2 % params.src_ne2;
    let s3 = i3 % params.src_ne3;

    let src_idx = params.offset_src +
                  s0 * params.stride_src0 +
                  s1 * params.stride_src1 +
                  s2 * params.stride_src2 +
                  s3 * params.stride_src3;

    let dst_idx = params.offset_dst +
                  i0 * params.stride_dst0 +
                  i1 * params.stride_dst1 +
                  i2 * params.stride_dst2 +
                  i3 * params.stride_dst3;

    dst[dst_idx] = src[src_idx];
}

#end(SHADER)
