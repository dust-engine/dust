#version 460

layout(location=0) in vec3 Vertex_Position;
layout(location=0) out vec3 vWorldPosition;
layout(set = 0, binding = 0) uniform Camera {
    mat4 ViewProj;
    mat4 RotationViewProj;
    vec3 position;
    float placeholder;
    vec3 forward;
    float placeholder2;
    float fov;
    float near;
    float far;
    float aspect_ratio;
};

void main() {
    vWorldPosition = Vertex_Position;// * bounding_box.w + bounding_box.xyz;
    gl_Position = RotationViewProj * vec4(vWorldPosition, 1.0);
}
