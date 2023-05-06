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



void main() {
    Block block = sbt.geometryInfo.blocks[gl_PrimitiveID];

    vec3 hitPointObject = gl_HitTEXT * gl_ObjectRayDirectionEXT + gl_ObjectRayOriginEXT;
    vec3 offsetInBox = vec3(hitAttributes.voxelId >> 4, (hitAttributes.voxelId >> 2) & 3, hitAttributes.voxelId & 3);
    vec3 boxCenterObject = block.position.xyz + offsetInBox + vec3(0.5);
    vec3 normalObject = CubedNormalize(hitPointObject - boxCenterObject);

    int8_t faceId = int8_t(normalObject.x) * int8_t(3) + int8_t(normalObject.y) * int8_t(2) + int8_t(normalObject.z);
    uint8_t faceIdU = uint8_t(min((faceId > 0 ? (faceId-1) : (6 + faceId)), 5));

    IrradianceCacheFace hashEntry = sbt.irradianceCache.entries[gl_PrimitiveID].faces[faceIdU];
    uint16_t lastAccessedFrameIndex = sbt.irradianceCache.entries[gl_PrimitiveID].lastAccessedFrameIndex[faceIdU];

    // irradiance, pre multiplied with albedo
    f16vec3 irradiance = hashEntry.irradiance * float16_t(pow(0.999, uint16_t(pushConstants.frameIndex) - lastAccessedFrameIndex));
    f16vec3 radiance = irradiance / float16_t(bitCount(uint(hashEntry.mask)));



    vec3 current = imageLoad(u_imgOutput, ivec2(gl_LaunchIDEXT.xy)).xyz;
    current += radiance * 0.02;
    imageStore(u_imgOutput, ivec2(gl_LaunchIDEXT.xy), vec4(current, 1.0));
}
