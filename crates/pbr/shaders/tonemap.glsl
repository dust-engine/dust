#version 460
#extension GL_GOOGLE_include_directive : require

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, rgba16f) uniform readonly image2D hdrInput;
layout(set = 0, binding = 1, rgba16f)   uniform writeonly image2D ldrOutput;
layout(set = 0, binding = 2, rgba8)   uniform readonly image2D sdrInput;

layout(set = 0, binding = 3, std140) uniform LpmCtl {
    uvec4 ctl[24];
} lpm;

layout(push_constant) uniform Push {
    float exposure;
} pc;

#define A_GPU  1
#define A_GLSL 1

#include "ffx_a.h"

AU4 LpmFilterCtl(AU1 i) { return lpm.ctl[i]; }

#define LPM_NO_SETUP 1
#include "ffx_lpm.h"

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 extent = imageSize(hdrInput);
    if (coord.x >= extent.x || coord.y >= extent.y) return;

    vec4 sdr = imageLoad(sdrInput, coord);
    if (sdr.a == 1.0) {
        imageStore(ldrOutput, coord, sdr);
        return;
    }

    vec4 hdr = imageLoad(hdrInput, coord);
    vec4 outColor = vec4(0.0, 0.0, 0.0, 1.0);
    if (hdr.a > 0.0) {
        vec3 c = hdr.rgb * pc.exposure;
        c = max(c, vec3(0.0));
        LpmFilter(c.r, c.g, c.b,
                  /*shoulder*/  false,
                  /*con*/       false,
                  /*soft*/      false,
                  /*con2*/      false,
                  /*clip*/      false,
                  /*scaleOnly*/ false);
        outColor.rgb = c * (1.0 - sdr.a) + sdr.rgb * sdr.a;
    }
    imageStore(ldrOutput, coord, outColor);
}
