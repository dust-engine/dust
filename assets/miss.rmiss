
#version 460
#extension GL_EXT_ray_tracing : require

layout(set = 0, binding = 0) uniform writeonly image2D u_imgOutput;

void main() {
    imageStore(u_imgOutput, ivec2(gl_LaunchIDEXT.xy), vec4(0.1, 0.1, 0.1, 1.0));
}
