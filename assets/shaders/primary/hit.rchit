#include "../headers/standard.glsl"
#include "../headers/sbt.glsl"
#include "../headers/normal.glsl"
#include "../headers/nrd.glsl"
#include "../headers/layout.playout"
#include "../headers/color.glsl"

#ifdef DEBUG_VISUALIZE_SPATIAL_HASH
#include "../headers/spatial_hash.glsl"
#endif

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

    vec3 normalObject = CubedNormalize(hitPointObject - boxCenterObject);
    vec3 normalWorld = gl_ObjectToWorldEXT * vec4(normalObject, 0.0);


    uint8_t palette_index = uint8_t(0);
    #ifdef DEBUG_VISUALIZE_SPATIAL_HASH
    vec3 boxCenterWorld = gl_ObjectToWorldEXT * vec4(boxCenterObject, 1.0);
    SpatialHashKey key;
    key.position = ivec3((boxCenterWorld / 4.0));
    key.direction = normal2FaceID(normalWorld);

    vec3 indirect_radiance; // The amount of incoming radiance at the voxel
    uint sample_count;
    bool found = SpatialHashGet(key, indirect_radiance, sample_count);
    vec4 packed = REBLUR_FrontEnd_PackRadianceAndNormHitDist(indirect_radiance, 0.0);
    imageStore(img_illuminance, ivec2(gl_LaunchIDEXT.xy), packed);

    uint32_t packed_albedo = block.avg_albedo;
    vec4 albedo = vec4(
        float((packed_albedo >> 22) & 1023) / 1023.0,
        float((packed_albedo >> 12) & 1023) / 1023.0,
        float((packed_albedo >> 2) & 1023) / 1023.0,
        float(packed_albedo & 3) / 3.0
    );

    imageStore(img_albedo, ivec2(gl_LaunchIDEXT.xy), albedo);
    #else
    
    imageStore(img_illuminance, ivec2(gl_LaunchIDEXT.xy), vec4(0.0));
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

    imageStore(img_albedo, ivec2(gl_LaunchIDEXT.xy), vec4(albedo, 1.0));
    #endif


    // Store the contribution from photon maps
    imageStore(img_depth, ivec2(gl_LaunchIDEXT.xy), vec4(gl_HitTEXT));

    imageStore(img_normal, ivec2(gl_LaunchIDEXT.xy), NRD_FrontEnd_PackNormalAndRoughness(normalWorld, 1.0, float(palette_index)));


    // Saved: | 8 bit voxel id | 8 bit palette_index | 16 bit instance id |
    uint voxel_id_info = (uint(hitAttributes.voxelId) << 24) | uint(gl_InstanceID & 0xFFFF) | (uint(palette_index) << 16);
    imageStore(img_voxel_id, ivec2(gl_LaunchIDEXT.xy), uvec4(voxel_id_info, 0, 0, 0));

    vec3 hitPointWorld = gl_HitTEXT * gl_WorldRayDirectionEXT + gl_WorldRayOriginEXT;
    vec3 hitPointModel = gl_WorldToObjectEXT * vec4(hitPointWorld, 1.0);
    vec4 hitPointWorldLastFrameH = instances[gl_InstanceID] * vec4(hitPointModel, 1.0);
    vec3 hitPointWorldLastFrame = hitPointWorldLastFrameH.xyz / hitPointWorldLastFrameH.w;
    imageStore(img_motion, ivec2(gl_LaunchIDEXT.xy), vec4(hitPointWorldLastFrame - hitPointWorld, 0.0));
}
