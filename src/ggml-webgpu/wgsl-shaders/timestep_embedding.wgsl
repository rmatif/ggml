@group(0) @binding(0)
var<storage, read_write> src: array<f32>;

@group(0) @binding(1)
var<storage, read_write> dst: array<f32>;

struct Params {
    offset_src: u32, // in elements
    offset_dst: u32, // in elements
    stride_dst1: u32, // in elements
    n_timestep: u32,
    dim: u32,
    max_period: u32
};

@group(0) @binding(2)
var<uniform> params: Params;

override wg_size: u32;

@compute @workgroup_size(wg_size)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = wid.x;
    if (i >= params.n_timestep) {
        return;
    }

    let half_dim = params.dim / 2u;
    let dst_row = params.offset_dst + i * params.stride_dst1;

    if ((params.dim & 1u) == 1u && lid.x == 0u) {
        dst[dst_row + 2u * half_dim] = 0.0;
    }

    if (half_dim == 0u) {
        return;
    }

    let timestep = src[params.offset_src + i];
    let log_max_period = log(f32(params.max_period));

    var j = lid.x;
    while (j < half_dim) {
        let freq = exp(-log_max_period * f32(j) / f32(half_dim));
        let arg = timestep * freq;
        dst[dst_row + j] = cos(arg);
        dst[dst_row + j + half_dim] = sin(arg);
        j += wg_size;
    }
}
