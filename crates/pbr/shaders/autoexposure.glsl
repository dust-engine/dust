#version 460

// Auto-exposure metering. Dispatched as a single 256-thread workgroup once per
// frame, after lighting is complete and before the denoise/upscale dispatch.
//
// Computes the geometric mean (log-average) luminance of the HDR input over a
// decimated sample lattice, converts it to an exposure scale that maps the
// scene's average to 18% mid-gray, and adapts toward that target over time.
// The result is written to a 1x1 R16F image consumed by three readers that
// must agree on one value:
//   - MetalFX `exposureTexture` / DLSS-RR `pInExposureTexture` (the upscaler
//     multiplies input color by it to understand displayed brightness),
//   - the tonemap pass, which multiplies the denoised HDR by it before LPM.

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, rgba16f) uniform readonly image2D hdrInput;
// Read-write: holds the adapted exposure scale from the previous frame.
layout(set = 0, binding = 1, r16f) uniform image2D exposureImage;

layout(set = 0, binding = 2, std140) uniform AutoExposureCtl {
    // Seconds since the previous frame; drives the adaptation rate.
    float dt;
    // Exposure compensation in stops, applied on top of the metered value.
    float ev_compensation;
    // Adaptation rate in 1/seconds (`alpha = 1 - exp(-dt * speed)`).
    float adaptation_speed;
    // Nonzero on the first frame after target (re)creation: the exposure image
    // holds undefined data, so adopt the metered target directly.
    uint first_frame;
    // Per-sample log2-luminance clamp, bounding the influence of outliers
    // (direct sun disk above, UI-adjacent black below).
    float min_log_luminance;
    float max_log_luminance;
} ctl;

shared float sh_sum[256];
shared uint sh_count[256];

void main() {
    uint tid = gl_LocalInvocationID.x;
    ivec2 extent = imageSize(hdrInput);

    // Decimate to a lattice of at most ~256x256 samples: statistically ample
    // for scene-average metering at a fraction of a full-image reduction.
    ivec2 stride = max(extent / 256, ivec2(1));
    ivec2 lattice = (extent + stride - 1) / stride;
    uint total = uint(lattice.x * lattice.y);

    float sum = 0.0;
    uint count = 0u;
    for (uint i = tid; i < total; i += 256u) {
        ivec2 cell = ivec2(int(i) % lattice.x, int(i) / lattice.x);
        ivec2 coord = min(cell * stride, extent - 1);
        vec4 hdr = imageLoad(hdrInput, coord);
        // Alpha 0 marks egui-occluded pixels (no scene radiance); skip them so
        // menus don't drag the meter toward black.
        if (hdr.a == 0.0) {
            continue;
        }
        float lum = dot(max(hdr.rgb, vec3(0.0)), vec3(0.2126, 0.7152, 0.0722));
        sum += clamp(log2(max(lum, 1e-8)), ctl.min_log_luminance, ctl.max_log_luminance);
        count += 1u;
    }
    sh_sum[tid] = sum;
    sh_count[tid] = count;
    barrier();

    for (uint offset = 128u; offset > 0u; offset >>= 1u) {
        if (tid < offset) {
            sh_sum[tid] += sh_sum[tid + offset];
            sh_count[tid] += sh_count[tid + offset];
        }
        barrier();
    }

    if (tid != 0u) {
        return;
    }

    float prev = imageLoad(exposureImage, ivec2(0)).r;
    bool has_prev = ctl.first_frame == 0u && prev > 0.0 && !isinf(prev);
    if (sh_count[0] == 0u) {
        // Fully occluded frame: nothing to meter, hold the previous exposure.
        if (!has_prev) {
            imageStore(exposureImage, ivec2(0), vec4(1.0));
        }
        return;
    }

    float avg_luminance = exp2(sh_sum[0] / float(sh_count[0]));
    // Scale mapping the metered average to 18% mid-gray, biased by EV
    // compensation.
    float target = 0.18 * exp2(ctl.ev_compensation) / avg_luminance;

    float value;
    if (has_prev) {
        // Exponential adaptation in log space, so brightening and darkening
        // converge symmetrically in stops.
        float alpha = 1.0 - exp(-ctl.dt * ctl.adaptation_speed);
        value = prev * pow(target / prev, alpha);
    } else {
        value = target;
    }
    // Keep well inside R16F range.
    value = clamp(value, exp2(-14.0), exp2(14.0));
    imageStore(exposureImage, ivec2(0), vec4(value));
}
