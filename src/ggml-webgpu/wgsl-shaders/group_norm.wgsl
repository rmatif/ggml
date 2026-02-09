#define(VARIANTS)

[
  {
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_SUFFIX": "inplace",
    "DECLS": ["INPLACE"]
  }
]

#end(VARIANTS)

#define(DECLS)

#decl(NOT_INPLACE)

fn update(src_offset: u32, dst_offset: u32, mean: f32, scale: f32) {
    dst[dst_offset] = (src[src_offset] - mean) * scale;
}

@group(0) @binding(1)
var<storage, read_write> dst: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

#enddecl(NOT_INPLACE)

#decl(INPLACE)

fn update(src_offset: u32, dst_offset: u32, mean: f32, scale: f32) {
    src[dst_offset] = (src[src_offset] - mean) * scale;
}

@group(0) @binding(1)
var<uniform> params: Params;

#enddecl(INPLACE)

#end(DECLS)

#define(SHADER)

@group(0) @binding(0)
var<storage, read_write> src: array<f32>;

struct Params {
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

    // Shape of src/dst
    ne0: u32,
    ne1: u32,
    ne2: u32,
    ne3: u32,

    n_groups: u32,
    eps: f32,
};

DECLS

override wg_size: u32;
var<workgroup> scratch_sum: array<f32, wg_size>;
var<workgroup> scratch_sumsq: array<f32, wg_size>;

@compute @workgroup_size(wg_size)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    if (params.n_groups == 0u || params.ne2 == 0u) {
        return;
    }

    // One workgroup per (batch, group).
    let group_batch = wid.x;
    let group_idx = group_batch % params.n_groups;
    let i3 = group_batch / params.n_groups;
    if (i3 >= params.ne3) {
        return;
    }

    let channels_per_group = (params.ne2 + params.n_groups - 1u) / params.n_groups;
    let start_ch = group_idx * channels_per_group;
    let end_ch = min(start_ch + channels_per_group, params.ne2);
    if (end_ch <= start_ch) {
        return;
    }

    let step = end_ch - start_ch;
    let elems_per_channel = params.ne0 * params.ne1;
    let group_elems = elems_per_channel * step;

    var sum = 0.0f;
    var idx = lid.x;
    while (idx < group_elems) {
        let ch_rel = idx / elems_per_channel;
        let rem = idx - ch_rel * elems_per_channel;
        let i1 = rem / params.ne0;
        let i0 = rem - i1 * params.ne0;
        let i2 = start_ch + ch_rel;
        let src_idx = params.offset_src + i0 * params.stride_src0 + i1 * params.stride_src1 +
                      i2 * params.stride_src2 + i3 * params.stride_src3;
        sum += src[src_idx];
        idx += wg_size;
    }
    scratch_sum[lid.x] = sum;
    workgroupBarrier();

    var offset = wg_size / 2u;
    while (offset > 0u) {
        if (lid.x < offset) {
            scratch_sum[lid.x] += scratch_sum[lid.x + offset];
        }
        offset = offset / 2u;
        workgroupBarrier();
    }

    let mean = scratch_sum[0] / f32(group_elems);

    var sumsq = 0.0f;
    idx = lid.x;
    while (idx < group_elems) {
        let ch_rel = idx / elems_per_channel;
        let rem = idx - ch_rel * elems_per_channel;
        let i1 = rem / params.ne0;
        let i0 = rem - i1 * params.ne0;
        let i2 = start_ch + ch_rel;
        let src_idx = params.offset_src + i0 * params.stride_src0 + i1 * params.stride_src1 +
                      i2 * params.stride_src2 + i3 * params.stride_src3;
        let d = src[src_idx] - mean;
        sumsq += d * d;
        idx += wg_size;
    }
    scratch_sumsq[lid.x] = sumsq;
    workgroupBarrier();

    offset = wg_size / 2u;
    while (offset > 0u) {
        if (lid.x < offset) {
            scratch_sumsq[lid.x] += scratch_sumsq[lid.x + offset];
        }
        offset = offset / 2u;
        workgroupBarrier();
    }

    let variance = scratch_sumsq[0] / f32(group_elems);
    let scale = inverseSqrt(variance + params.eps);

    idx = lid.x;
    while (idx < group_elems) {
        let ch_rel = idx / elems_per_channel;
        let rem = idx - ch_rel * elems_per_channel;
        let i1 = rem / params.ne0;
        let i0 = rem - i1 * params.ne0;
        let i2 = start_ch + ch_rel;
        let src_idx = params.offset_src + i0 * params.stride_src0 + i1 * params.stride_src1 +
                      i2 * params.stride_src2 + i3 * params.stride_src3;
        let dst_idx = params.offset_dst + i0 * params.stride_dst0 + i1 * params.stride_dst1 +
                      i2 * params.stride_dst2 + i3 * params.stride_dst3;
        update(src_idx, dst_idx, mean, scale);
        idx += wg_size;
    }
}

#end(SHADER)
