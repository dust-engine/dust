#version 460

layout(location=0) in vec3 Vertex_Position;
layout(location=0) out vec3 vWorldPosition;

void main() {
    vWorldPosition = Vertex_Position;// * bounding_box.w + bounding_box.xyz;
    gl_Position = vec4(Vertex_Position, 1.0);
    //gl_Position = ViewProj * vec4(vWorldPosition, 1.0);
}
