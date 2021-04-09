#version 460

layout(location=0) in vec3 Vertex_Position;
layout(set = 0, binding = 0) uniform Camera {
    mat4 ViewProj;
    mat4 RotationViewProj;
    mat4 Proj;
};

void main() {
    gl_Position = RotationViewProj * vec4(Vertex_Position, 1.0);
}
