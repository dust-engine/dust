#version 460
#include "standard.glsl"

layout(shaderRecordEXT) buffer Sbt {
    GeometryInfo geometryInfo;
    MaterialInfo materialInfo;
    PaletteInfo paletteInfo;
    IrradianceCache irradianceCache;
} sbt;

hitAttributeEXT HitAttribute {
    uint8_t voxelId;
} hitAttributes;
layout(location = 0) rayPayloadInEXT struct RayPayload {
    vec3 illuminance;
} payload;




void main() {
    Block block = sbt.geometryInfo.blocks[gl_PrimitiveID];

    vec3 hitPointObject = gl_HitTEXT * gl_ObjectRayDirectionEXT + gl_ObjectRayOriginEXT;
    vec3 offsetInBox = vec3(hitAttributes.voxelId >> 4, (hitAttributes.voxelId >> 2) & 3, hitAttributes.voxelId & 3);
    vec3 boxCenterObject = block.position.xyz + offsetInBox + vec3(0.5);
    vec3 normalObject = CubedNormalize(hitPointObject - boxCenterObject);

    int8_t faceId = int8_t(normalObject.x) * int8_t(3) + int8_t(normalObject.y) * int8_t(2) + int8_t(normalObject.z);
    uint8_t faceIdU = uint8_t(min((faceId > 0 ? (faceId-1) : (6 + faceId)), 5));

    IrradianceCacheFace hashEntry = sbt.irradianceCache.entries[gl_PrimitiveID].faces[faceIdU];
    if (hashEntry.mask == 0) {
        return;
    }
    uint16_t lastAccessedFrameIndex = sbt.irradianceCache.entries[gl_PrimitiveID].lastAccessedFrameIndex[faceIdU];

    // irradiance, pre multiplied with albedo.
    vec3 irradiance = hashEntry.irradiance * pow(RETENTION_FACTOR, uint16_t(pushConstants.frameIndex) - lastAccessedFrameIndex);

    float scaling_factors =
    // Divide by the activated surface area of the voxel
    (1.0 / float(bitCount(uint(hashEntry.mask)))) *
    // Correction based on the retention factor.
    /// \sigma a^n from 0 to inf is 1 / (1 - a).
    (1.0 - RETENTION_FACTOR)
    ;
    vec3 radiance = irradiance * scaling_factors;


    imageStore(u_illuminance, ivec2(gl_LaunchIDEXT.xy), vec4(payload.illuminance + radiance, 1.0));
}
