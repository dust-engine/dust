#include "../headers/standard.glsl"
#include "../headers/layout.playout"
#include "../headers/sbt.glsl"
#include "../headers/normal.glsl"
#include "../headers/spatial_hash.glsl"
#include "../headers/color.glsl"
#include "../headers/surfel.glsl"

hitAttributeEXT HitAttribute {
    uint8_t voxelId;
} hitAttributes;

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


// Light leaves the surfel (S) and hit another voxel (V).
// We sample V and find out the outgoing radiance from the spatial hash.
void main() {
    Block block = sbt.geometryInfo.blocks[gl_PrimitiveID];
    vec3 aabbCenterObject = block.position.xyz;
    
    
    if (hitAttributes.voxelId == uint8_t(0xFF)) {
        // Ray hits the rough model.
        aabbCenterObject += 2.0;
    } else {
        // Ray hits the fine model.
        vec3 offsetInBox = vec3(hitAttributes.voxelId >> 4, (hitAttributes.voxelId >> 2) & 3, hitAttributes.voxelId & 3);
        aabbCenterObject += offsetInBox + vec3(0.5);
    }
    
    vec3 hitPointObject = gl_HitTEXT * gl_ObjectRayDirectionEXT + gl_ObjectRayOriginEXT;
    vec3 normalObject = hitPointObject - aabbCenterObject;
    vec3 normalWorld = CubedNormalize(gl_ObjectToWorldEXT * vec4(normalObject, 0.0));

    aabbCenterObject = block.position.xyz + 2.0;
    vec3 aabbCenterWorld = gl_ObjectToWorldEXT * vec4(aabbCenterObject, 1.0);
    
    SpatialHashKey key;
    key.position = ivec3((aabbCenterWorld / 4.0));
    key.direction = normal2FaceID(normalWorld);

    vec3 radiance;
    uint sample_count = 0;
    // Sample the hit location in the spatial hash
    bool found = SpatialHashGet(key, radiance, sample_count);

    uvec2 noiseSampleLocation;
    uint width = textureSize(blue_noise[0], 0).x;
    noiseSampleLocation.y = gl_LaunchIDEXT.x / width;
    noiseSampleLocation.x = gl_LaunchIDEXT.x - noiseSampleLocation.y * width;
    float rand = texelFetch(blue_noise[0], ivec2((noiseSampleLocation + uvec2(114, 40) + push_constants.rand) % textureSize(blue_noise[0], 0)), 0).x;
    if (found) {
        // Stocasticlly select one voxel from the block
        #ifdef SHADER_INT_64
        uint numVoxelInAabb = GridNumVoxels(block.mask);
        #else
        uint numVoxelInAabb = GridNumVoxels(u32vec2(block.mask1, block.mask2));
        #endif
        float randomVoxelIndexFloat = mix(0.0, float(numVoxelInAabb), rand);
        uint randomVoxelIndex = max(uint(randomVoxelIndexFloat), numVoxelInAabb - 1);

        // Use the albedo of the randomly selected voxel as the albedo
        uint8_t palette_index = sbt.materialInfo.materials[block.material_ptr + randomVoxelIndex];
        u8vec4 color = sbt.paletteInfo.palette[palette_index];
        vec3 albedo = color.xyz / 255.0;
        albedo.x = SRGBToLinear(albedo.x);
        albedo.y = SRGBToLinear(albedo.y);
        albedo.z = SRGBToLinear(albedo.z);
        albedo = SRGBToXYZ(albedo);


        // Add to surfel value
        SurfelEntry surfel = surfel_pool[gl_LaunchIDEXT.x];

        SpatialHashKey key;
        key.position = ivec3((surfel.position / 4.0));
        key.direction = uint8_t(surfel.direction);
        // `radiance` is the incoming radiance at the hit location.
        // Spatial hash stores the incoming radiance. The incoming radiance at the
        // surfel location is the outgoing radiance at the hit location.
        SpatialHashInsert(key, radiance * albedo);
        // TODO: Optimize the Insert / Get flow so they only do the loop once.

        // TODO: Maybe add more samples to the patch: need heuristic.
    } else {
        // TODO: enqueue the patch stocastically
        // TODO: Also need to enqueue a sample of strength = 0
        // TODO: weight expotential fallout over time
        float probability_to_schedule = 1.0 / float(sample_count + 2);
        if (rand > probability_to_schedule) {
            uint index = gl_LaunchIDEXT.x % SurfelPoolSize;


            SurfelEntry entry;
            entry.position = aabbCenterWorld;
            entry.direction = normal2FaceID(normalWorld);
            surfel_pool[index] = entry;
        }
    }
}



// Surfel sampling strategy:
// - Surfel should be sampled porportional to their visual importance.
// - Visual importance is the amount of light that eventually reaches the eye.

// We may change the surfel being sampled when
// - The final gather ray hits the surfel
// - The surfel ray hits the surfel


// How to estimate the visual importance of a surfel?
// if we turn the eye into a light source, the visual importance is the amount of radiance received
// Russian roulette. When we trace from the eye, we +1 on each bounce.
// Then we terminate the ray with probability porportional to the albedo.

// At the end of this, we collect all the +1s . We then select 2000 of those
// with importance sampling based on the number of +1s.

//  However, more importantly, the sample rate should be porportional to
// the variance.



