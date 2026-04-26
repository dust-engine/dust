#version 460
#extension GL_GOOGLE_include_directive : require

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, rgba16f) uniform readonly image2D hdrInput;
layout(set = 0, binding = 1, rgba16f)   uniform writeonly image2D ldrOutput;
layout(set = 0, binding = 2, rgba8)   uniform readonly image2D sdrInput;

layout(set = 0, binding = 3, std140) uniform LpmCtl {
    uvec4 ctl[24];
    vec3 sdr_mapping_col0;
    uint gamma_mode;
    vec3 sdr_mapping_col1;
    // Bitfield selecting the LpmFilter config path for the active preset.
    // Must match `pbr::tonemap::lpm_flags` in the Rust side.
    //   bit 0 = shoulder
    //   bit 1 = con
    //   bit 2 = soft
    //   bit 3 = con2
    //   bit 4 = clip
    //   bit 5 = scaleOnly
    uint lpm_flags;
    vec3 sdr_mapping_col2;
} lpm;

#define LPM_FLAG_SHOULDER   1u
#define LPM_FLAG_CON        2u
#define LPM_FLAG_SOFT       4u
#define LPM_FLAG_CON2       8u
#define LPM_FLAG_CLIP      16u
#define LPM_FLAG_SCALEONLY 32u

#define A_GPU  1
#define A_GLSL 1

#include "ffx_a.h"

AU4 LpmFilterCtl(AU1 i) { return lpm.ctl[i]; }

#define LPM_NO_SETUP 1
#include "ffx_lpm.h"

vec3 LinearToSRGB(vec3 color) {
    // https://registry.khronos.org/DataFormat/specs/1.3/dataformat.1.3.html#TRANSFER_SRGB
    // Approximately pow(color, 1.0 / 2.2)
    return mix(1.055 * pow(color, vec3(1.0 / 2.4)) - 0.055, 12.92 * color, lessThan(color, vec3(0.0031308)));
}
vec3 LinearToSCRGB(vec3 color) {
    // https://registry.khronos.org/DataFormat/specs/1.3/dataformat.1.3.html#TRANSFER_SRGB
    return mix(LinearToSRGB(color), -1.055 * pow(-color, vec3(1.0 / 2.4)) + 0.055, lessThan(color, vec3(-0.0031308)));
}

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 extent = imageSize(hdrInput);
    if (coord.x >= extent.x || coord.y >= extent.y) return;

    vec4 outColor = imageLoad(sdrInput, coord);
    if (outColor.a != 0.0) {
        // SDR has output
        outColor.rgb = mat3(lpm.sdr_mapping_col0, lpm.sdr_mapping_col1, lpm.sdr_mapping_col2) * outColor.rgb;
    }
    if (outColor.a != 1.0) {
        vec4 hdr = imageLoad(hdrInput, coord);
        vec3 c = max(hdr.rgb, vec3(0.0));
        bool shoulder  = (lpm.lpm_flags & LPM_FLAG_SHOULDER)   != 0u;
        bool con       = (lpm.lpm_flags & LPM_FLAG_CON)        != 0u;
        bool soft      = (lpm.lpm_flags & LPM_FLAG_SOFT)       != 0u;
        bool con2      = (lpm.lpm_flags & LPM_FLAG_CON2)       != 0u;
        bool clip      = (lpm.lpm_flags & LPM_FLAG_CLIP)       != 0u;
        bool scaleOnly = (lpm.lpm_flags & LPM_FLAG_SCALEONLY)  != 0u;
        LpmFilter(c.r, c.g, c.b, shoulder, con, soft, con2, clip, scaleOnly);
        outColor.rgb = c * (1.0 - outColor.a) + outColor.rgb * outColor.a;
    }

    switch (lpm.gamma_mode) {
        case 0: // Linear
            break;
        case 1: // sRGB
            outColor.rgb = LinearToSRGB(outColor.rgb);
            break;
        case 2: // scRGB
            outColor.rgb = LinearToSCRGB(outColor.rgb);
            break;
        case 3: // DCI_P3
            // https://registry.khronos.org/DataFormat/specs/1.3/dataformat.1.3.html#TRANSFER_DCIP3
            outColor.rgb = pow((outColor.rgb / 52.37), vec3(1 / 2.6));
            break;
        case 4: // DisplayP3
            // https://vkdoc.net/man/VkColorSpaceKHR
            outColor.rgb = mix(
                1.055 * pow(outColor.rgb, vec3(1.0 / 2.4)) - 0.055,
                12.92 * outColor.rgb,
                lessThan(outColor.rgb, vec3(0.0030186))
            );
            break;
        case 5: // ITU
            // ITU OETF
            // https://registry.khronos.org/DataFormat/specs/1.3/dataformat.1.3.html#TRANSFER_ITU
            const float beta = 0.0181;
            const float alpha = 1.0993;
            outColor.rgb = mix(
                alpha * pow(outColor.rgb, vec3(0.45)) - (alpha - 1.0),
                4.5 * outColor.rgb,
                lessThan(outColor.rgb, vec3(beta))
            );
            break;
        case 6: // ST2084_PQ
            // https://registry.khronos.org/DataFormat/specs/1.3/dataformat.1.3.html#TRANSFER_PQ
            const float m1 = 2610.0 / 16384.0;
            const float m2 = 2523.0 / 4096.0 * 128.0;
            const float c1 = 3424.0 / 4096.0;
            const float c2 = 2413.0 / 4096.0 * 32.0;
            const float c3 = 2392.0 / 4096.0 * 32.0;
            vec3 Ym1 = pow(outColor.rgb, vec3(m1));
            outColor.rgb = pow((c1 + c2 * Ym1) / (1.0 + c3 * Ym1), vec3(m2));
            break;
        case 7:
            // HLG OETF
            // https://registry.khronos.org/DataFormat/specs/1.3/dataformat.1.3.html#TRANSFER_HLG
            const float a = 0.17883277;
            const float b = 1.0 - 4.0 * a;
            const float c = 0.55991073;
            outColor.rgb = mix(
                a * log(12.0 * outColor.rgb - b) + c,
                sqrt(3 * outColor.rgb),
                lessThan(outColor.rgb, vec3(1.0 / 12.0))
            );
            break;
        case 8:
            // AdobeRGB
            // https://registry.khronos.org/DataFormat/specs/1.3/dataformat.1.3.html#TRANSFER_ADOBERGB
            outColor.rgb = pow(outColor.rgb, vec3(256.0 / 563.0));
            break;
        default:
            break;
    }
    imageStore(ldrOutput, coord, outColor);
}
