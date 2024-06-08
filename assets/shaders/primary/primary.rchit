#include "../headers/header.glsl"
#include "../headers/layout.playout"

#include "../headers/sbt.glsl"
#include "../headers/color.glsl"


hitAttributeEXT HitAttribute {
    uint voxelId;
} hitAttributes;

void main() {
    Block block = sbt.geometryInfo.blocks[gl_PrimitiveID];
    
    // Calculate nexthit location
    vec3 hitPointObject = gl_HitTEXT * gl_ObjectRayDirectionEXT + gl_ObjectRayOriginEXT;
    vec3 offsetInBox = vec3(hitAttributes.voxelId >> 4, (hitAttributes.voxelId >> 2) & 3, hitAttributes.voxelId & 3);

    #ifdef DEBUG_VISUALIZE_SPATIAL_HASH
    vec3 boxCenterObject = block.position.xyz + vec3(2.0);
    #else
    vec3 boxCenterObject = block.position.xyz + offsetInBox + vec3(0.5);
    #endif

    uint8_t palette_index = uint8_t(0);
    
    // Sample the albedo from the voxel
    #ifdef SHADER_INT_64
    u32vec2 masked = unpack32(block.mask & ((uint64_t(1) << hitAttributes.voxelId) - 1));
    uint32_t voxelMemoryOffset = bitCount(masked.x) + bitCount(masked.y);
    #else
    u32vec2 masked = u32vec2(
        hitAttributes.voxelId < 32 ? block.mask1 & ((1 << hitAttributes.voxelId) - 1) : block.mask1,
        hitAttributes.voxelId >= 32 ? block.mask2 & ((1 << (hitAttributes.voxelId - 32)) - 1) : 0
    );
    uint32_t voxelMemoryOffset = uint32_t(bitCount(masked.x) + bitCount(masked.y));
    #endif

    palette_index = sbt.materialInfo.materials[block.material_ptr + voxelMemoryOffset];
    u8vec4 color = sbt.paletteInfo.palette[palette_index];

    vec3 albedo = color.xyz / 255.0;

    albedo.x = SRGBToLinear(albedo.x);
    albedo.y = SRGBToLinear(albedo.y);
    albedo.z = SRGBToLinear(albedo.z);

    imageStore(img_illuminance, ivec2(gl_LaunchIDEXT.xy), vec4(albedo, 1.0));
}
