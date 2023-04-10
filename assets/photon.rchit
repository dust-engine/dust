#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_atomic_float : require
#extension GL_EXT_samplerless_texture_functions: require

layout(set = 0, binding = 0) uniform writeonly image2D u_imgOutput;
layout(set = 0, binding = 2) uniform texture2D blue_noise;
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
    vec3 normal;
};
layout(location = 0) rayPayloadInEXT PhotonRayPayload photon;


struct PhotonEnergy {
    vec3 energy;
    uint lastAccessedFrame;
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

layout(push_constant) uniform PushConstants {
    // Indexed by block id
    uint rand;
    uint frameIndex;
} pushConstants;

hitAttributeEXT vec3 normal;


void main() {
    const uint lastAccessedFrame = atomicExchange(sbt.photon_energy_info.blocks[gl_PrimitiveID].lastAccessedFrame, pushConstants.frameIndex);

    const uint frameDifference = pushConstants.frameIndex - lastAccessedFrame;

    if (frameDifference > 0) {
        vec3 prevEnergy = sbt.photon_energy_info.blocks[gl_PrimitiveID].energy;
        vec3 nextEnergy = prevEnergy * pow(0.9, frameDifference * 0.1) + photon.energy;

        sbt.photon_energy_info.blocks[gl_PrimitiveID].energy = nextEnergy;
    }
    if (frameDifference == 0) {
        atomicAdd(sbt.photon_energy_info.blocks[gl_PrimitiveID].energy.x, photon.energy.x);
        atomicAdd(sbt.photon_energy_info.blocks[gl_PrimitiveID].energy.y, photon.energy.y);
        atomicAdd(sbt.photon_energy_info.blocks[gl_PrimitiveID].energy.z, photon.energy.z);
    }

    // To make A syncronizes-with B, we need
    // A and B are in the instance of each other's memory scopes
    // A and B are in the instance of C's execution scope

    photon.energy *= 0.2;
    photon.hitT = gl_HitTEXT;
    photon.normal = normal;

    //vec3 noiseSample = texelFetch(blue_noise, ivec2((gl_LaunchIDEXT.xy + uvec2(12, 24) + pushConstants.rand) % textureSize(blue_noise, 0)), 0).xyz;
    //float d = dot(noiseSample, normal);
    //if (d < 0.0) {
    //    noiseSample = -noiseSample;
    //}
    //photon.origin = photon.origin + photon.dir * (gl_HitTEXT * 0.99);
    //photon.dir = noiseSample;
    // What we're doing here:
    // atomic exchange. one thread gets the old frame index, all other threads get the new frame index.
    // multiplication. one thread multiplies the old energy by 0.5, all other threads do nothing
    // addition. all threads add energy.
}
// TODO: use atomic swap on frame index, use weighted average function for irradiance cache
// This makes photon mapping nearly free by avoiding resets and just storing a timestamp
