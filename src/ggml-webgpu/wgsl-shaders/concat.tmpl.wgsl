#define(VARIANTS)

[
  {
    "REPLS": {
      "TYPE": "f32"
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

@group(0) @binding(0)
var<storage, read_write> src0: array<{{TYPE}}>;

@group(0) @binding(1)
var<storage, read_write> src1: array<{{TYPE}}>;

@group(0) @binding(2)
var<storage, read_write> dst: array<{{TYPE}}>;

struct Params {
    ne: u32,
    offset_src0: u32, // in elements
    offset_src1: u32, // in elements
    offset_dst: u32,  // in elements

    src0_ne0: u32,
    src0_ne1: u32,
    src0_ne2: u32,
    src0_ne3: u32,
    src0_nb0: u32, // in elements
    src0_nb1: u32,
    src0_nb2: u32,
    src0_nb3: u32,

    src1_ne0: u32,
    src1_ne1: u32,
    src1_ne2: u32,
    src1_ne3: u32,
    src1_nb0: u32, // in elements
    src1_nb1: u32,
    src1_nb2: u32,
    src1_nb3: u32,

    dst_ne0: u32,
    dst_ne1: u32,
    dst_ne2: u32,
    dst_ne3: u32,
    dst_nb0: u32, // in elements
    dst_nb1: u32,
    dst_nb2: u32,
    dst_nb3: u32,

    dim: u32
};

@group(0) @binding(3)
var<uniform> params: Params;

override wg_size: u32;
@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.ne) {
        return;
    }

    var i = idx;
    let i3 = i / (params.dst_ne2 * params.dst_ne1 * params.dst_ne0);
    i = i % (params.dst_ne2 * params.dst_ne1 * params.dst_ne0);
    let i2 = i / (params.dst_ne1 * params.dst_ne0);
    i = i % (params.dst_ne1 * params.dst_ne0);
    let i1 = i / params.dst_ne0;
    let i0 = i % params.dst_ne0;

    let dst_idx = params.offset_dst +
                  i0 * params.dst_nb0 +
                  i1 * params.dst_nb1 +
                  i2 * params.dst_nb2 +
                  i3 * params.dst_nb3;

    let is_src0 = i0 < params.src0_ne0 &&
                  i1 < params.src0_ne1 &&
                  i2 < params.src0_ne2 &&
                  i3 < params.src0_ne3;

    if (is_src0) {
        let src0_idx = params.offset_src0 +
                       i0 * params.src0_nb0 +
                       i1 * params.src0_nb1 +
                       i2 * params.src0_nb2 +
                       i3 * params.src0_nb3;
        dst[dst_idx] = src0[src0_idx];
        return;
    }

    var j0 = i0;
    var j1 = i1;
    var j2 = i2;
    var j3 = i3;

    if (params.dim == 0u) {
        j0 = i0 - params.src0_ne0;
    } else if (params.dim == 1u) {
        j1 = i1 - params.src0_ne1;
    } else if (params.dim == 2u) {
        j2 = i2 - params.src0_ne2;
    } else {
        j3 = i3 - params.src0_ne3;
    }

    let src1_idx = params.offset_src1 +
                   j0 * params.src1_nb0 +
                   j1 * params.src1_nb1 +
                   j2 * params.src1_nb2 +
                   j3 * params.src1_nb3;
    dst[dst_idx] = src1[src1_idx];
}

#end(SHADER)
