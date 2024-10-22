#include "../headers/header.glsl"
#include "../headers/layout.playout"

#include "../headers/sbt.glsl"
#include "../headers/color.glsl"


hitAttributeEXT HitAttribute {
    uint voxelId;
} hitAttributes;

void main() {
    imageStore(img_illuminance, ivec2(gl_LaunchIDEXT.xy), vec4(1.0, 1.0, 1.0, 1.0));
    return;
    Block block = sbt.geometryInfo.blocks[gl_PrimitiveID];
    
    // Calculate nexthit location
    vec3 hitPointObject = gl_HitTEXT * gl_ObjectRayDirectionEXT + gl_ObjectRayOriginEXT;
    vec3 offsetInBox = vec3(hitAttributes.voxelId >> 6, (hitAttributes.voxelId >> 3) & 7, hitAttributes.voxelId & 7);

    vec3 boxCenterObject = (block.min + block.max) / 2.0;

    uint8_t palette_index = uint8_t(0);
    
    // Sample the albedo from the voxel
    uint32_t voxelMemoryOffset = GridCountOnesBefore(block.mask, hitAttributes.voxelId);

    palette_index = sbt.materialInfo.materials[block.material_ptr + voxelMemoryOffset];
    u8vec4 color = sbt.paletteInfo.palette[palette_index-1];

    vec3 albedo = color.xyz / 255.0;

    albedo.x = SRGBToLinear(albedo.x);
    albedo.y = SRGBToLinear(albedo.y);
    albedo.z = SRGBToLinear(albedo.z);

    imageStore(img_illuminance, ivec2(gl_LaunchIDEXT.xy), vec4(albedo, 1.0));
}
