#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

layout(set = 0, binding = 0) uniform writeonly image2D u_imgOutput;
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
struct PhotonEnergy {
    vec3 energy;
    uint count;
};

layout(buffer_reference) buffer PhotonEnergyInfo {
    // Indexed by block id
    PhotonEnergy blocks[];
};


layout(shaderRecordEXT) buffer sbt {
    GeometryInfo geometryInfo;
    MaterialInfo materialInfo;
    PaletteInfo paletteInfo;
    PhotonEnergyInfo photon_energy_info;
};
//layout(set = 1, binding = 1, r8ui) uniform readonly uimage2D bindless_StorageImage[];

hitAttributeEXT uint8_t voxelId;

vec3 randomColorList[5] = {
    vec3(0.976, 0.906, 0.906),
    vec3(0.871, 0.839, 0.839),
    vec3(0.824, 0.796, 0.796),
    vec3(0.678, 0.627, 0.651),
    vec3(0.49, 0.576, 0.541),
};

void main() {
    Block block = geometryInfo.blocks[gl_PrimitiveID];

    u32vec2 masked = unpack32(block.mask & ((uint64_t(1) << voxelId) - 1));
    uint32_t voxelMemoryOffset = bitCount(masked.x) + bitCount(masked.y);

    uint8_t palette_index = materialInfo.materials[block.material_ptr + voxelMemoryOffset];
    u8vec4 color = paletteInfo.palette[palette_index];

    PhotonEnergy energy = photon_energy_info.blocks[gl_PrimitiveID];
    imageStore(u_imgOutput, ivec2(gl_LaunchIDEXT.xy), vec4(energy.energy / float(energy.count) / 100.0, 1.0));

    //imageStore(u_imgOutput, ivec2(gl_LaunchIDEXT.xy), vec4(vec3(color.rgb) / 255.0, 1.0));
}
