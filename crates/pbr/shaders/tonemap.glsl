#version 460
#extension GL_GOOGLE_include_directive : require

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, rgba16f) uniform readonly image2D hdrInput;
layout(set = 0, binding = 1, rgba16f)   uniform writeonly image2D ldrOutput;
layout(set = 0, binding = 2, rgba8)   uniform readonly image2D sdrInput;

layout(set = 0, binding = 3, std140) uniform LpmCtl {
    uvec4 ctl[24];
    uint gamma_mode;
} lpm;

#define A_GPU  1
#define A_GLSL 1

#include "ffx_a.h"

AU4 LpmFilterCtl(AU1 i) { return lpm.ctl[i]; }

#define LPM_NO_SETUP 1
#include "ffx_lpm.h"



float LinearToSRGB(float color) {
    // Approximately pow(color, 1.0 / 2.2)
    return color <= 0.0031308 ? 12.92 * color : 1.055 * pow(color, 1.0 / 2.4) - 0.055;
}
float LinearToSCRGB(float color) {
  return color <= -0.0031308 ? -1.055 * pow(-color, 1.0 / 2.4) + 0.055 : LinearToSRGB(color);
}
float LinearToDisplayP3(float color)
{
    return color < 0.0030186 ? 12.92 * color : 1.055 * pow(color, 1.0 / 2.4) - 0.055;
}
float LinearToITU(float color) {
  const float beta = 0.0181;
  const float alpha = 1.0993;
  return color < beta ? 4.5 * color : alpha * pow(color, 0.45) - (alpha - 1.0);
}

float LinearToHLG(float color) {
  const float a = 0.17883277;
  const float b = 1.0 - 4.0 * a;
  const float c = 0.55991073;
  return color < (1.0 / 12.0) ? sqrt(3 * color) : a * log(12.0 * color - b) + c;
}

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 extent = imageSize(hdrInput);
    if (coord.x >= extent.x || coord.y >= extent.y) return;

    vec4 outColor = imageLoad(sdrInput, coord);
    if (outColor.a != 1.0) {
        vec4 hdr = imageLoad(hdrInput, coord);
        vec3 c = max(hdr.rgb, vec3(0.0));
        LpmFilter(c.r, c.g, c.b,
                /*shoulder*/  false,
                /*con*/       false,
                /*soft*/      false,
                /*con2*/      false,
                /*clip*/      false,
                /*scaleOnly*/ false);
        outColor.rgb = c * (1.0 - outColor.a) + outColor.rgb * outColor.a;
    }

    switch (lpm.gamma_mode) {
        case 0: // IDENTITY
            break;
        case 1: // sRGB
            outColor.r = LinearToSRGB(outColor.r);
            outColor.g = LinearToSRGB(outColor.g);
            outColor.b = LinearToSRGB(outColor.b);
            break;
        default:
            break;
    }
    imageStore(ldrOutput, coord, outColor);
}
