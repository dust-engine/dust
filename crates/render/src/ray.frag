#version 460

layout(location=0) out vec4 f_color;
layout(location=0) in vec3 vWorldPosition;

void main() {
    f_color = vec4(vWorldPosition, 1.0);
}
