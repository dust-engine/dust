#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_nonuniform_qualifier : require

struct RayPayload {
    vec3 color;
};


layout(shaderRecordEXT) buffer sbt {
    uint64_t geometryInfo;
    uint32_t materialInfo;
};
layout(set = 1, binding = 1, r8ui) uniform readonly uimage2D bindless_StorageImage[];

vec3 randomColorList[5] = {
    vec3(0.976, 0.906, 0.906),
    vec3(0.871, 0.839, 0.839),
    vec3(0.824, 0.796, 0.796),
    vec3(0.678, 0.627, 0.651),
    vec3(0.49, 0.576, 0.541),
};

layout(location = 0) rayPayloadInEXT RayPayload primaryRayPayload;

void main() {
    float v = gl_RayTmaxEXT / 400.0;
    primaryRayPayload.color = vec3(v,v,v);
}
