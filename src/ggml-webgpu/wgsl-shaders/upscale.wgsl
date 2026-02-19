@group(0) @binding(0)
var<storage, read_write> src: array<f32>;

@group(0) @binding(1)
var<storage, read_write> dst: array<f32>;

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

    sf0: f32,
    sf1: f32,
    sf2: f32,
    sf3: f32,
    pixel_offset: f32,

    mode_flags: u32
};

@group(0) @binding(2)
var<uniform> params: Params;

const SCALE_MODE_NEAREST: u32  = 0u;
const SCALE_MODE_BILINEAR: u32 = 1u;
const SCALE_MODE_BICUBIC: u32  = 2u;

const SCALE_FLAG_ANTIALIAS: u32 = (1u << 9u);

const BCOEFFS1: vec4<f32> = vec4(1.25, -2.25, 0.0, 1.0);
const BCOEFFS2: vec4<f32> = vec4(-0.75, 3.75, -6.0, 3.0);

fn clamp_i32(v: i32, lo: i32, hi: i32) -> i32 {
    return min(max(v, lo), hi);
}

fn powers(x: f32) -> vec4<f32> {
    return vec4(x * x * x, x * x, x, 1.0);
}

fn bicubic(p0: f32, p1: f32, p2: f32, p3: f32, x: f32) -> f32 {
    return p0 * dot(BCOEFFS2, powers(x + 1.0)) +
           p1 * dot(BCOEFFS1, powers(x)) +
           p2 * dot(BCOEFFS1, powers(1.0 - x)) +
           p3 * dot(BCOEFFS2, powers(2.0 - x));
}

fn triangle_filter(x: f32) -> f32 {
    return max(1.0 - abs(x), 0.0);
}

fn src_index(i0: u32, i1: u32, i2: u32, i3: u32) -> u32 {
    return params.offset_src +
           i0 * params.stride_src0 +
           i1 * params.stride_src1 +
           i2 * params.stride_src2 +
           i3 * params.stride_src3;
}

fn dst_index(i0: u32, i1: u32, i2: u32, i3: u32) -> u32 {
    return params.offset_dst +
           i0 * params.stride_dst0 +
           i1 * params.stride_dst1 +
           i2 * params.stride_dst2 +
           i3 * params.stride_dst3;
}

fn map_index(i: u32, sf: f32, src_ne: u32) -> u32 {
    let mapped = i32(f32(i) / sf);
    let clamped = clamp_i32(mapped, 0, i32(src_ne) - 1);
    return u32(clamped);
}

fn fetch_clamped(i0: i32, i1: i32, i2: u32, i3: u32) -> f32 {
    let x = u32(clamp_i32(i0, 0, i32(params.src_ne0) - 1));
    let y = u32(clamp_i32(i1, 0, i32(params.src_ne1) - 1));
    return src[src_index(x, y, i2, i3)];
}

fn interpolate_nearest(i0: u32, i1: u32, i2: u32, i3: u32) -> f32 {
    let i00 = map_index(i0, params.sf0, params.src_ne0);
    let i01 = map_index(i1, params.sf1, params.src_ne1);
    let i02 = map_index(i2, params.sf2, params.src_ne2);
    let i03 = map_index(i3, params.sf3, params.src_ne3);
    return src[src_index(i00, i01, i02, i03)];
}

fn interpolate_bilinear(i0: u32, i1: u32, i2: u32, i3: u32) -> f32 {
    let i02 = map_index(i2, params.sf2, params.src_ne2);
    let i03 = map_index(i3, params.sf3, params.src_ne3);

    let y = (f32(i1) + params.pixel_offset) / params.sf1 - params.pixel_offset;
    let y0 = clamp_i32(i32(floor(y)), 0, i32(params.src_ne1) - 1);
    let y1 = clamp_i32(y0 + 1, 0, i32(params.src_ne1) - 1);
    let dy = clamp(y - f32(y0), 0.0, 1.0);

    let x = (f32(i0) + params.pixel_offset) / params.sf0 - params.pixel_offset;
    let x0 = clamp_i32(i32(floor(x)), 0, i32(params.src_ne0) - 1);
    let x1 = clamp_i32(x0 + 1, 0, i32(params.src_ne0) - 1);
    let dx = clamp(x - f32(x0), 0.0, 1.0);

    let a = src[src_index(u32(x0), u32(y0), i02, i03)];
    let b = src[src_index(u32(x1), u32(y0), i02, i03)];
    let c = src[src_index(u32(x0), u32(y1), i02, i03)];
    let d = src[src_index(u32(x1), u32(y1), i02, i03)];
    return a * (1.0 - dx) * (1.0 - dy) +
           b * dx * (1.0 - dy) +
           c * (1.0 - dx) * dy +
           d * dx * dy;
}

fn interpolate_bilinear_antialias(i0: u32, i1: u32, i2: u32, i3: u32) -> f32 {
    let i02 = map_index(i2, params.sf2, params.src_ne2);
    let i03 = map_index(i3, params.sf3, params.src_ne3);

    let support1 = max(1.0, 1.0 / params.sf1);
    let invscale1 = 1.0 / support1;
    let support0 = max(1.0, 1.0 / params.sf0);
    let invscale0 = 1.0 / support0;

    let y = (f32(i1) + params.pixel_offset) / params.sf1;
    let x = (f32(i0) + params.pixel_offset) / params.sf0;

    let x_min = max(i32(x - support0 + params.pixel_offset), 0);
    let x_max = min(i32(x + support0 + params.pixel_offset), i32(params.src_ne0));
    let y_min = max(i32(y - support1 + params.pixel_offset), 0);
    let y_max = min(i32(y + support1 + params.pixel_offset), i32(params.src_ne1));

    var val = 0.0;
    var total_weight = 0.0;
    for (var sy = y_min; sy < y_max; sy++) {
        let weight_y = triangle_filter((f32(sy) - y + params.pixel_offset) * invscale1);
        for (var sx = x_min; sx < x_max; sx++) {
            let weight_x = triangle_filter((f32(sx) - x + params.pixel_offset) * invscale0);
            let weight = weight_x * weight_y;
            if (weight <= 0.0) {
                continue;
            }

            let pixel = src[src_index(u32(sx), u32(sy), i02, i03)];
            val += pixel * weight;
            total_weight += weight;
        }
    }

    if (total_weight > 0.0) {
        val /= total_weight;
    }
    return val;
}

fn interpolate_bicubic(i0: u32, i1: u32, i2: u32, i3: u32) -> f32 {
    let i02 = map_index(i2, params.sf2, params.src_ne2);
    let i03 = map_index(i3, params.sf3, params.src_ne3);

    let coord = (vec2<f32>(f32(i0), f32(i1)) + params.pixel_offset) / vec2(params.sf0, params.sf1) - params.pixel_offset;
    let dx = fract(coord.x);
    let dy = fract(coord.y);
    let x0 = i32(floor(coord.x));
    let y0 = i32(floor(coord.y));

    return bicubic(
        bicubic(fetch_clamped(x0 - 1, y0 - 1, i02, i03), fetch_clamped(x0, y0 - 1, i02, i03), fetch_clamped(x0 + 1, y0 - 1, i02, i03), fetch_clamped(x0 + 2, y0 - 1, i02, i03), dx),
        bicubic(fetch_clamped(x0 - 1, y0 + 0, i02, i03), fetch_clamped(x0, y0 + 0, i02, i03), fetch_clamped(x0 + 1, y0 + 0, i02, i03), fetch_clamped(x0 + 2, y0 + 0, i02, i03), dx),
        bicubic(fetch_clamped(x0 - 1, y0 + 1, i02, i03), fetch_clamped(x0, y0 + 1, i02, i03), fetch_clamped(x0 + 1, y0 + 1, i02, i03), fetch_clamped(x0 + 2, y0 + 1, i02, i03), dx),
        bicubic(fetch_clamped(x0 - 1, y0 + 2, i02, i03), fetch_clamped(x0, y0 + 2, i02, i03), fetch_clamped(x0 + 1, y0 + 2, i02, i03), fetch_clamped(x0 + 2, y0 + 2, i02, i03), dx),
        dy);
}

override wg_size: u32;
@compute @workgroup_size(wg_size)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(num_workgroups) num_wg: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
    let idx = (wg_id.y * num_wg.x + wg_id.x) * wg_size + local_id.x;
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

    let mode = params.mode_flags & 0xFFu;

    var value: f32 = 0.0;
    switch (mode) {
        case SCALE_MODE_NEAREST: {
            value = interpolate_nearest(i0, i1, i2, i3);
        }
        case SCALE_MODE_BILINEAR: {
            if ((params.mode_flags & SCALE_FLAG_ANTIALIAS) != 0u) {
                value = interpolate_bilinear_antialias(i0, i1, i2, i3);
            } else {
                value = interpolate_bilinear(i0, i1, i2, i3);
            }
        }
        case SCALE_MODE_BICUBIC: {
            value = interpolate_bicubic(i0, i1, i2, i3);
        }
        default: {
            value = 0.0;
        }
    }

    dst[dst_index(i0, i1, i2, i3)] = value;
}
