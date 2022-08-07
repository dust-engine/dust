#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

struct RayPayload {
    vec3 color;
};

struct Block
{
    u16vec4 position;
    uint64_t mask;
    uint32_t material_ptr;
    uint32_t reserved;
};

layout(buffer_reference, buffer_reference_align = 8, scalar) buffer GeometryInfo {
    Block blocks[];
};
layout(buffer_reference) buffer MaterialInfo {
    uint8_t materials[];
};

layout(shaderRecordEXT) buffer sbt {
    GeometryInfo geometryInfo;
    MaterialInfo materialInfo;
};
layout(set = 1, binding = 1, r8ui) uniform readonly uimage2D bindless_StorageImage[];

hitAttributeEXT uint8_t voxelId;

vec3 randomColorList[5] = {
    vec3(0.976, 0.906, 0.906),
    vec3(0.871, 0.839, 0.839),
    vec3(0.824, 0.796, 0.796),
    vec3(0.678, 0.627, 0.651),
    vec3(0.49, 0.576, 0.541),
};

layout(location = 0) rayPayloadInEXT RayPayload primaryRayPayload;

void main() {
    uint32_t material_ptr = geometryInfo.blocks[gl_PrimitiveID+7].material_ptr;
    uint32_t palette_value = material_ptr + uint32_t(voxelId);
    primaryRayPayload.color = randomColorList[uint32_t(voxelId) % 5];
}
