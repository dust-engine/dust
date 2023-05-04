#version 460
#include "standard.glsl"

layout(push_constant) uniform PushConstants {
    // Indexed by block id
    uint rand;
    uint frameIndex;
} pushConstants;

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
    
    // Calculate nexthit location
    vec3 hitPointObject = gl_HitTEXT * gl_ObjectRayDirectionEXT + gl_ObjectRayOriginEXT;
    vec3 offsetInBox = vec3(hitAttributes.voxelId >> 4, (hitAttributes.voxelId >> 2) & 3, hitAttributes.voxelId & 3);
    vec3 boxCenterObject = block.position.xyz + offsetInBox + vec3(0.5);
    vec3 normalObject = CubedNormalize(hitPointObject - boxCenterObject);
    vec3 normalWorld = gl_ObjectToWorldEXT * vec4(normalObject, 0.0);

    u32vec2 masked = unpack32(block.mask & ((uint64_t(1) << hitAttributes.voxelId) - 1));
    uint32_t voxelMemoryOffset = bitCount(masked.x) + bitCount(masked.y);

    uint8_t palette_index = sbt.materialInfo.materials[block.material_ptr + voxelMemoryOffset];
    u8vec4 color = sbt.paletteInfo.palette[palette_index];


    vec3 albedo = color.xyz / 255.0;

    // Store the contribution from photon maps
    imageStore(u_depth, ivec2(gl_LaunchIDEXT.xy), vec4(gl_HitTEXT));
    imageStore(u_normal, ivec2(gl_LaunchIDEXT.xy), vec4(normalWorld, 1.0));
    imageStore(u_albedo, ivec2(gl_LaunchIDEXT.xy), vec4(albedo, 1.0));
}
