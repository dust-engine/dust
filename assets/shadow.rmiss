
#version 460
#include "standard.glsl"

void main() {
    vec3 albedo = imageLoad(u_albedo, ivec2(gl_LaunchIDEXT.xy)).xyz;
    imageStore(u_imgOutput, ivec2(gl_LaunchIDEXT.xy), vec4(albedo, 1.0));
}
