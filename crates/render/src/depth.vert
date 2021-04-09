#version 460

layout(location=0) in vec3 Vertex_Position;
layout(set = 0, binding = 0) uniform Camera {
    mat4 ViewProj;
    mat4 RotationViewProj;
    vec3 CameraPosition;
    float placeholder;
    float fov;
    float near;
    float far;
    float aspect_ratio;
};

void main() {
    gl_Position = ViewProj * vec4(Vertex_Position, 1.0);
}
