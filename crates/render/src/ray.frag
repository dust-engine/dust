#version 460

layout(location=0) out vec4 f_color;
layout(location=0) in vec3 vWorldPosition;
layout(set = 0, binding = 0) uniform Camera {
    mat4 ViewProj;
    mat4 transform;
};
void main() {
    f_color = vec4(vWorldPosition, 1.0);
}
