#version 460
#include "standard.glsl"

layout(shaderRecordEXT) buffer Sbt {
    GeometryInfo geometryInfo;
    MaterialInfo materialInfo;
    PaletteInfo paletteInfo;
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


    uint8_t palette_index = sbt.materialInfo.materials[block.material_ptr + voxelMemoryOffset];
    u8vec4 color = sbt.paletteInfo.palette[palette_index];


    vec3 albedo = color.xyz / 255.0;
    albedo.x = SRGBToLinear(albedo.x);
    albedo.y = SRGBToLinear(albedo.y);
    albedo.z = SRGBToLinear(albedo.z);

    // Store the contribution from photon maps
    imageStore(u_depth, ivec2(gl_LaunchIDEXT.xy), vec4(gl_HitTEXT));

    // TODO: Is it actually normalWorld?
    imageStore(u_normal, ivec2(gl_LaunchIDEXT.xy), NRD_FrontEnd_PackNormalAndRoughness(normalWorld, 1.0, float(palette_index)));

    imageStore(u_albedo, ivec2(gl_LaunchIDEXT.xy), vec4(SRGBToXYZ(albedo), 1.0));

    // Saved: | 8 bit voxel id | 8 bit palette_index | 16 bit instance id |
    uint voxel_id_info = (uint(hitAttributes.voxelId) << 24) | uint(gl_InstanceID & 0xFFFF) | (uint(palette_index) << 16);
    imageStore(u_voxel_id, ivec2(gl_LaunchIDEXT.xy), uvec4(voxel_id_info, 0, 0, 0));

    vec3 hitPointWorld = gl_HitTEXT * gl_WorldRayDirectionEXT + gl_WorldRayOriginEXT;
    vec3 hitPointModel = gl_WorldToObjectEXT * vec4(hitPointWorld, 1.0);
    vec4 hitPointWorldLastFrameH = s_instances.last_frame_transforms[gl_InstanceID] * vec4(hitPointModel, 1.0);
    vec3 hitPointWorldLastFrame = hitPointWorldLastFrameH.xyz / hitPointWorldLastFrameH.w;
    imageStore(u_motion, ivec2(gl_LaunchIDEXT.xy), vec4(hitPointWorldLastFrame - hitPointWorld, 0.0));
}
