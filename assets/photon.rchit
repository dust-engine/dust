#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_atomic_float : require

layout(set = 0, binding = 0) uniform writeonly image2D u_imgOutput;
struct Block
{
    u16vec4 position;
    uint64_t mask;
    uint32_t material_ptr;
    uint32_t block_id;
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

struct PhotonRayPayload {
    vec3 energy;
    float hitT;
};
layout(location = 0) rayPayloadInEXT PhotonRayPayload photon;


struct PhotonEnergy {
    vec3 energy;
    uint count;
};

layout(buffer_reference) buffer PhotonEnergyInfo {
    // Indexed by block id
    PhotonEnergy blocks[];
};

layout(shaderRecordEXT) buffer Sbt {
    GeometryInfo geometryInfo;
    MaterialInfo materialInfo;
    PaletteInfo paletteInfo;
    PhotonEnergyInfo photon_energy_info;
} sbt;


void main() {
    atomicAdd(sbt.photon_energy_info.blocks[gl_PrimitiveID].count, 1);
    atomicAdd(sbt.photon_energy_info.blocks[gl_PrimitiveID].energy.x, photon.energy.x);
    atomicAdd(sbt.photon_energy_info.blocks[gl_PrimitiveID].energy.y, photon.energy.y);
    atomicAdd(sbt.photon_energy_info.blocks[gl_PrimitiveID].energy.z, photon.energy.z);


    photon.hitT = gl_HitTEXT;
}
