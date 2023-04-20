#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_atomic_float : require
#extension GL_EXT_samplerless_texture_functions: require

layout(set = 0, binding = 0) uniform writeonly image2D u_imgOutput;
layout(set = 0, binding = 3) uniform texture2D blue_noise;
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

layout(shaderRecordEXT) buffer Sbt {
    GeometryInfo geometryInfo;
    MaterialInfo materialInfo;
    PaletteInfo paletteInfo;
} sbt;

layout(push_constant) uniform PushConstants {
    // Indexed by block id
    uint rand;
    uint frameIndex;
} pushConstants;

hitAttributeEXT uint voxelId;

vec3 CubedNormalize(vec3 dir) {
    vec3 dir_abs = abs(dir);
    float max_element = max(dir_abs.x, max(dir_abs.y, dir_abs.z));
    return sign(dir) * step(max_element, dir_abs);
}



struct RadianceHashMapEntry {
    vec3 energy;
    uint32_t lastAccessedFrameIndex;
};
layout(set = 0, binding = 4) buffer RadianceHashMap {
    uint num_entries;
    RadianceHashMapEntry[] entries;
} radianceCache;

uvec3 pcg3d(uvec3 v)
{
    v = v * 1664525 + 1013904223;
    v.x += v.y*v.z;
    v.y += v.z*v.x;
    v.z += v.x*v.y;

    v = v ^ (v >> 16);
    v.x += v.y*v.z;
    v.y += v.z*v.x;
    v.z += v.x*v.y;
    return v;
}
uint hashPayload(
    uint instanceId,
    uint primitiveId,
    uint voxelId,
    uint faceId
) {
    uvec3 result = pcg3d(uvec3(instanceId, primitiveId, voxelId << 16 | faceId));
    return result.x + result.y + result.z;
}


void main() {
    if (photon.hitT != 0.0) {
        uint hash = hashPayload(gl_InstanceID, gl_PrimitiveID, voxelId, 0) % radianceCache.num_entries;
        const uint lastAccessedFrame = atomicExchange(radianceCache.entries[hash].lastAccessedFrameIndex, pushConstants.frameIndex);

        const uint frameDifference = pushConstants.frameIndex - lastAccessedFrame;

        if (frameDifference > 0) {
            vec3 prevEnergy = radianceCache.entries[hash].energy;
            vec3 nextEnergy = prevEnergy * pow(0.99, frameDifference) + photon.energy;

           radianceCache.entries[hash].energy = nextEnergy;
        }
        if (frameDifference == 0) {
            radianceCache.entries[hash].energy += photon.energy;
        }
    }
    
    Block block = sbt.geometryInfo.blocks[gl_PrimitiveID];
    u32vec2 blockMask = unpack32(block.mask);
    uint32_t numVoxelInBlock = bitCount(blockMask.x) + bitCount(blockMask.y);
    uint32_t randomVoxelId = pushConstants.rand % numVoxelInBlock;

    u32vec2 masked = unpack32(block.mask & ((uint64_t(1) << randomVoxelId) - 1));
    uint32_t voxelMemoryOffset = uint32_t(bitCount(masked) + bitCount(masked.y));

    uint8_t palette_index = sbt.materialInfo.materials[block.material_ptr + voxelMemoryOffset];
    u8vec4 color = sbt.paletteInfo.palette[palette_index];


    vec3 boxCenter = gl_ObjectToWorldEXT * vec4(block.position.xyz + vec3(2.0, 2.0, 2.0), 1.0);
    vec3 hitPoint = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    photon.normal = CubedNormalize(hitPoint - boxCenter);

    photon.energy *= vec3(color.xyz) / 255.0;
    photon.hitT = gl_HitTEXT;

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
