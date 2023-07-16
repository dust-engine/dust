#version 460
#include "standard.glsl"
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

hitAttributeEXT HitAttribute {
    uint8_t voxelId;
} hitAttributes;

void main() {
    Block block = sbt.geometryInfo.blocks[gl_PrimitiveID];

    {
        // Multiply ray energy by voxel albedo
        #ifdef SHADER_INT_64
        u32vec2 blockMask = unpack32(block.mask);
        uint32_t numVoxelInBlock = bitCount(blockMask.x) + bitCount(blockMask.y);
        #else
        uint32_t numVoxelInBlock = bitCount(block.mask1) + bitCount(block.mask2);
        #endif
        uint32_t randomVoxelId = pushConstants.rand % numVoxelInBlock;

        #ifdef SHADER_INT_64
        u32vec2 masked = unpack32(block.mask & ((uint64_t(1) << randomVoxelId) - 1));
        #else
        u32vec2 masked = u32vec2(
            randomVoxelId < 32 ? block.mask1 & ((1 << randomVoxelId) - 1) : block.mask1,
            randomVoxelId >= 32 ? block.mask2 & ((1 << (randomVoxelId - 32)) - 1) : 0
        );
        #endif
        uint32_t voxelMemoryOffset = uint32_t(bitCount(masked.x) + bitCount(masked.y));

        uint8_t palette_index = sbt.materialInfo.materials[block.material_ptr + voxelMemoryOffset];
        u8vec4 color = sbt.paletteInfo.palette[palette_index];

        vec3 albedo = color.xyz / 255.0;
        albedo.x = SRGBToLinear(albedo.x);
        albedo.y = SRGBToLinear(albedo.y);
        albedo.z = SRGBToLinear(albedo.z);
        albedo = SRGBToXYZ(albedo);

        photon.energy *= albedo;
    }

    // Calculate normal
    vec3 hitPointObject = gl_HitTEXT * gl_ObjectRayDirectionEXT + gl_ObjectRayOriginEXT;
    vec3 offsetInBox = vec3(hitAttributes.voxelId >> 4, (hitAttributes.voxelId >> 2) & 3, hitAttributes.voxelId & 3);
    vec3 boxCenterObject = block.position.xyz + offsetInBox + vec3(0.5);
    vec3 normalObject = CubedNormalize(hitPointObject - boxCenterObject);
    photon.normal = gl_ObjectToWorldEXT * vec4(normalObject, 0.0);

    int8_t faceId = int8_t(normalObject.x) * int8_t(3) + int8_t(normalObject.y) * int8_t(2) + int8_t(normalObject.z);
    uint8_t faceIdU = uint8_t(min((faceId > 0 ? (faceId-1) : (6 + faceId)), 5));
    // Accumulate energy
    const uint16_t lastAccessedFrame = sbt.irradianceCache.entries[gl_PrimitiveID].lastAccessedFrameIndex[faceIdU];
    sbt.irradianceCache.entries[gl_PrimitiveID].lastAccessedFrameIndex[faceIdU] = uint16_t(pushConstants.frameIndex);

    const uint16_t frameDifference = uint16_t(pushConstants.frameIndex) - lastAccessedFrame;

    vec3 strength = photon.energy *
    (1.0 - cos(sunlight_config.solar_intensity.w));

    vec3 prevEnergy = sbt.irradianceCache.entries[gl_PrimitiveID].faces[faceIdU].irradiance;
    if (frameDifference > 0) {
        vec3 nextEnergy = prevEnergy * pow(RETENTION_FACTOR, frameDifference) + strength;
        sbt.irradianceCache.entries[gl_PrimitiveID].faces[faceIdU].irradiance = f16vec3(nextEnergy);
    } else {
        sbt.irradianceCache.entries[gl_PrimitiveID].faces[faceIdU].irradiance = f16vec3(strength + prevEnergy);
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
    
    
    photon.hitT = gl_HitTEXT;
}
// TODO: use atomic swap on frame index, use weighted average function for irradiance cache
// This makes photon mapping nearly free by avoiding resets and just storing a timestamp
