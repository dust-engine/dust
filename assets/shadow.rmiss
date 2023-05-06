
#version 460
#include "standard.glsl"

void main() {
    float strength = 37.0; // FIXME: the light strength
    // TODO: multiply by incident angle cosine rule.
    vec3 albedo = imageLoad(u_albedo, ivec2(gl_LaunchIDEXT.xy)).xyz;
    imageStore(u_imgOutput, ivec2(gl_LaunchIDEXT.xy), vec4(albedo * strength, 1.0));
}
