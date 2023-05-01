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

struct IrradianceCacheFace {
    f16vec3 irradiance;
    uint16_t mask;
};
struct IrradianceCacheEntry {
    IrradianceCacheFace faces[6];
    uint16_t lastAccessedFrameIndex[6];
    uint32_t _reerved;
};
layout(buffer_reference, scalar) buffer IrradianceCache {
    IrradianceCacheEntry entries[];
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
    IrradianceCache irradianceCache;
} sbt;

layout(push_constant) uniform PushConstants {
    // Indexed by block id
    uint rand;
    uint frameIndex;
} pushConstants;
hitAttributeEXT HitAttribute {
    uint8_t voxelId;
} hitAttributes;

vec3 CubedNormalize(vec3 dir) {
    vec3 dir_abs = abs(dir);
    float max_element = max(dir_abs.x, max(dir_abs.y, dir_abs.z));
    return sign(dir) * step(max_element, dir_abs);
}

void main() {
    Block block = sbt.geometryInfo.blocks[gl_PrimitiveID];

    // Calculate normal
    vec3 hitPointObject = gl_HitTEXT * gl_ObjectRayDirectionEXT + gl_ObjectRayOriginEXT - block.position.xyz;
    vec3 offsetInBox = vec3(hitAttributes.voxelId >> 4, (hitAttributes.voxelId >> 2) & 3, hitAttributes.voxelId & 3);
    vec3 normalObject = CubedNormalize(hitPointObject - offsetInBox - vec3(0.5));
    photon.normal = gl_ObjectToWorldEXT * vec4(normalObject, 0.0);

    int8_t faceId = int8_t(normalObject.x) * int8_t(3) + int8_t(normalObject.y) * int8_t(2) + int8_t(normalObject.z);
    uint8_t faceIdU = uint8_t(min((faceId > 0 ? (faceId-1) : (6 + faceId)), 5));
    // Accumulate energy
    const uint16_t lastAccessedFrame = sbt.irradianceCache.entries[gl_PrimitiveID].lastAccessedFrameIndex[faceIdU];
    sbt.irradianceCache.entries[gl_PrimitiveID].lastAccessedFrameIndex[faceIdU] = uint16_t(pushConstants.frameIndex);

    const uint16_t frameDifference = uint16_t(pushConstants.frameIndex) - lastAccessedFrame;

    if (frameDifference > 0) {
        f16vec3 prevEnergy = sbt.irradianceCache.entries[gl_PrimitiveID].faces[faceIdU].irradiance;
        f16vec3 nextEnergy = prevEnergy * float16_t(pow(0.999, frameDifference)) + f16vec3(photon.energy);
        sbt.irradianceCache.entries[gl_PrimitiveID].faces[faceIdU].irradiance = nextEnergy;
    } else {
        sbt.irradianceCache.entries[gl_PrimitiveID].faces[faceIdU].irradiance += f16vec3(photon.energy);
    }

    // Calculate projected 2d hitpoint
    vec3 absNormal = abs(normalObject);
    vec2 hitPointSurface = vec2(
        mix(hitPointObject.y, hitPointObject.x, absNormal[1] + absNormal[2]),
        mix(hitPointObject.z, hitPointObject.y, absNormal[2])
    ); // range: 0-4
    u8vec2 hitPointCoords = min(u8vec2(floor(hitPointSurface)), u8vec2(3)); // range: 0, 1, 2, 3.
    uint8_t hitPointCoord = hitPointCoords.x * uint8_t(4) + hitPointCoords.y; // range: 0 - 15
    //sbt.irradianceCache.entries[gl_PrimitiveID].faces[hitAttributes.faceId].mask = uint16_t(1);
    sbt.irradianceCache.entries[gl_PrimitiveID].faces[faceIdU].mask |= uint16_t(1) << hitPointCoord;
    
    
    u32vec2 blockMask = unpack32(block.mask);
    uint32_t numVoxelInBlock = bitCount(blockMask.x) + bitCount(blockMask.y);
    uint32_t randomVoxelId = pushConstants.rand % numVoxelInBlock;

    u32vec2 masked = unpack32(block.mask & ((uint64_t(1) << randomVoxelId) - 1));
    uint32_t voxelMemoryOffset = uint32_t(bitCount(masked) + bitCount(masked.y));

    uint8_t palette_index = sbt.materialInfo.materials[block.material_ptr + voxelMemoryOffset];
    u8vec4 color = sbt.paletteInfo.palette[palette_index];

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
