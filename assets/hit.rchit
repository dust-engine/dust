#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

layout(set = 0, binding = 0) uniform writeonly image2D u_imgOutput;
layout(set = 0, binding = 1) uniform writeonly image2D u_diffuseOutput;
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
    uint lastAccessedFrame;
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
layout(location = 0) rayPayloadInEXT float hitT;

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

    vec3 diffuseColor = vec3(color) / 255.0;
    // The parameter 0.01 was derived from the 0.99 retention factor. It's not arbitrary.
    vec3 indirectContribution = 0.01 * energy.energy * diffuseColor;

    // Store the contribution from photon maps
    imageStore(u_imgOutput, ivec2(gl_LaunchIDEXT.xy), vec4(indirectContribution + diffuseColor * 0.2, 1.0));
    imageStore(u_diffuseOutput, ivec2(gl_LaunchIDEXT.xy), vec4(diffuseColor, 1.0));
    hitT = gl_HitTEXT;
}
