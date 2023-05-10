
#version 460
#include "standard.glsl"
layout(location = 0) rayPayloadInEXT struct RayPayload {
    f16vec3 color;
    f16vec3 albedo;
} payload;



void main() {
    const float ambient_light = 3.8;
    // TODO: calculate ambient light, add into main texture. We assume that the ambient light is 0.1.
    imageStore(u_imgOutput, ivec2(gl_LaunchIDEXT.xy), vec4(payload.color + ambient_light * payload.albedo, 1.0));
}
