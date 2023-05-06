
#version 460
#include "standard.glsl"

void main() {
    // TODO: calculate ambient light, add into main texture.
    
    vec3 current = imageLoad(u_imgOutput, ivec2(gl_LaunchIDEXT.xy)).xyz;
    current += vec3(0.1);
    imageStore(u_imgOutput, ivec2(gl_LaunchIDEXT.xy), vec4(current, 1.0));
}
