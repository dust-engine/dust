#version 460

layout(location=0) out vec4 f_color;
layout(location=0) in vec3 vWorldPosition;

void main() {
    f_color = vec4(0.5, 0.7, 0.3, 1.0);
}
