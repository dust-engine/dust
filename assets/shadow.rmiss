
#version 460
#extension GL_EXT_ray_tracing : require

layout(set = 0, binding = 0, rgba32f) uniform image2D u_imgOutput;
layout(set = 0, binding = 1, rgba32f) uniform readonly image2D u_diffuseColor;
void main() {
    vec3 environmentColor = imageLoad(u_imgOutput, ivec2(gl_LaunchIDEXT.xy)).xyz;
    vec3 diffuseColor = imageLoad(u_diffuseColor, ivec2(gl_LaunchIDEXT.xy)).xyz;

    vec3 mixedColor = diffuseColor * 0.5;
    imageStore(u_imgOutput, ivec2(gl_LaunchIDEXT.xy), vec4(environmentColor, 1.0));
}
