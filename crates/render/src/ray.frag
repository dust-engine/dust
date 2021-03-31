#version 460
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require


layout (constant_id = 0) const uint MAX_ITERATION_VALUE = 1000;
const vec4 bounding_box = vec4(0.0, 0.0, 0.0, 16.0);

layout(location=0) out vec4 f_color;
layout(location=0) in vec3 vWorldPosition;
layout(set = 0, binding = 0) uniform Camera {
    mat4 ViewProj;
    mat4 transform;
};
layout(set = 1, binding = 0) readonly buffer Chunk {
    uint8_t testdata[];
};


void main() {
    f_color = vec4(vWorldPosition, 1.0);
}
