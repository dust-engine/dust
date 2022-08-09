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
layout(buffer_reference, buffer_reference_align = 1, scalar) buffer MaterialInfo {
    uint8_t materials[];
};
layout(buffer_reference) buffer PaletteInfo {
    u8vec4 palette[];
};

layout(shaderRecordEXT) buffer sbt {
    GeometryInfo geometryInfo;
    MaterialInfo materialInfo;
    PaletteInfo paletteInfo;
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
    uint32_t material_ptr = geometryInfo.blocks[gl_PrimitiveID].material_ptr;

    uint8_t palette_index = materialInfo.materials[material_ptr + uint32_t(voxelId)];
    u8vec4 color = paletteInfo.palette[palette_index];
    primaryRayPayload.color = vec3(color.r / 255.0, color.g / 255.0, color.b / 255.0);
}
