
#include "../headers/standard.glsl"
#include "../headers/sbt.glsl"
#include "../headers/normal.glsl"
#include "../headers/layout.playout"
#include "../headers/spatial_hash.glsl"
#include "../headers/surfel.glsl"
#include "../headers/color.glsl"
#include "../headers/nrd.glsl"


hitAttributeEXT HitAttribute {
    uint _unused;
} hitAttributes;
layout(location = 0) rayPayloadInEXT struct RayPayload {
    vec3 illuminance;
} payload;

#ifdef SHADER_INT_64
#define GridType uint64_t
uint GridNumVoxels(GridType grid) {
    u32vec2 unpacked = unpack32(grid);
    return bitCount(masked.x) + bitCount(masked.y);
}
#else
#define GridType u32vec2
uint GridNumVoxels(GridType grid) {
    return bitCount(grid.x) + bitCount(grid.y);
}
#endif

// The ray goes from the primary hit point to the secondary hit point.
// We sample the spatial hash at the secondary hit point, find out
// the outgoing radiance, and store into the radiance texture.
void main() {
    Block block = sbt.geometryInfo.blocks[gl_PrimitiveID];

    vec3 aabbCenterObject = block.position.xyz + 2.0;
    vec3 hitPointObject = gl_HitTEXT * gl_ObjectRayDirectionEXT + gl_ObjectRayOriginEXT;
    vec3 aabbNormalObject = hitPointObject - aabbCenterObject; // normal of the larger voxels
    vec3 aabbNormalWorld = CubedNormalize(gl_ObjectToWorldEXT * vec4(aabbNormalObject, 0.0));
    vec3 aabbCenterWorld = gl_ObjectToWorldEXT * vec4(aabbCenterObject, 1.0); // Center of the larger voxels
    

    SpatialHashKey key;
    key.position = ivec3((aabbCenterWorld / 4.0));
    key.direction = normal2FaceID(aabbNormalWorld);

    vec3 indirect_radiance;
    uint sample_count;
    bool found = SpatialHashGet(key, indirect_radiance, sample_count);
    float probability_to_schedule = 1.0 / float(sample_count + 2);
    float noise_sample = texelFetch(blue_noise[0], ivec2((gl_LaunchIDEXT.xy + uvec2(34, 21) + push_constants.rand) % textureSize(blue_noise[0], 0)), 0).x;

    if (noise_sample > probability_to_schedule) {
        uint index = gl_LaunchIDEXT.x + gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x;
        index = index % SurfelPoolSize;

        SurfelEntry entry;
        entry.position = aabbCenterWorld;
        entry.direction = normal2FaceID(aabbNormalWorld);
        surfel_pool[index] = entry;
    }

    // indirect radiance is the incoming radiance at the secondary hit location.
    // Multiply with albedo to convert into outgoing radiance at secondary hit location.

    #ifdef SHADER_INT_64
    uint numVoxelInAabb = GridNumVoxels(block.mask);
    #else
    uint numVoxelInAabb = GridNumVoxels(u32vec2(block.mask1, block.mask2));
    #endif
    float rand = texelFetch(blue_noise[0], ivec2((gl_LaunchIDEXT.xy + uvec2(18, 74) + push_constants.rand) % textureSize(blue_noise[0], 0)), 0).x;
    float randomVoxelIndexFloat = mix(0.0, float(numVoxelInAabb), rand);
    uint randomVoxelIndex = max(uint(randomVoxelIndexFloat), numVoxelInAabb - 1);

    // Convert into albedo
    uint8_t palette_index = sbt.materialInfo.materials[block.material_ptr + randomVoxelIndex];
    u8vec4 color = sbt.paletteInfo.palette[palette_index];
    vec3 albedo = color.xyz / 255.0;
    albedo.x = SRGBToLinear(albedo.x);
    albedo.y = SRGBToLinear(albedo.y);
    albedo.z = SRGBToLinear(albedo.z);

    indirect_radiance = sRGB2AECScg(AECScg2sRGB(indirect_radiance) * albedo);
    
    vec3 value = payload.illuminance;
    #ifdef CONTRIBUTION_SECONDARY_SPATIAL_HASH
    value += indirect_radiance;
    #endif
    vec4 packed = REBLUR_FrontEnd_PackRadianceAndNormHitDist(value, gl_HitTEXT);

    #ifndef DEBUG_VISUALIZE_SPATIAL_HASH
    imageStore(img_illuminance, ivec2(gl_LaunchIDEXT.xy), packed);
    #endif
}

// TODO: final gather and surfel should both use the corser grid.
