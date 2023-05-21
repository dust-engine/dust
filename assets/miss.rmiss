
#version 460
#include "standard.glsl"

void main() {
    imageStore(u_illuminance, ivec2(gl_LaunchIDEXT.xy), vec4(0.5, 0.1, 0.1, 1.0));
    imageStore(u_albedo, ivec2(gl_LaunchIDEXT.xy), vec4(1.0));
}
