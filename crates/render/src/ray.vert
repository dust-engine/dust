#version 460

layout(location=0) in vec3 Vertex_Position;
layout(location=0) out vec3 vWorldPosition;
layout(set = 0, binding = 0) uniform Camera {
    mat4 RotationViewProj;
    mat4 transform;
};

void main() {
    vWorldPosition = Vertex_Position;// * bounding_box.w + bounding_box.xyz;
    gl_Position = RotationViewProj * vec4(vWorldPosition, 1.0);
}
