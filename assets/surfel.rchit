#version 460
#include "standard.glsl"

layout(shaderRecordEXT) buffer Sbt {
    GeometryInfo geometryInfo;
    MaterialInfo materialInfo;
    PaletteInfo paletteInfo;
} sbt;

void main() {
    Block block = sbt.geometryInfo.blocks[gl_PrimitiveID];
    vec3 aabbCenterObject = block.position.xyz + 2.0;
    
    vec3 hitPointObject = gl_HitTEXT * gl_ObjectRayDirectionEXT + gl_ObjectRayOriginEXT;
    vec3 normalObject = hitPointObject - aabbCenterObject;
    vec3 normalWorld = CubedNormalize(gl_ObjectToWorldEXT * vec4(normalObject, 0.0));

    vec3 aabbCenterWorld = gl_ObjectToWorldEXT * vec4(aabbCenterObject, 1.0);
    

    SpatialHashKey key;
    key.position = ivec3(round(aabbCenterWorld / 4.0));
    key.direction = normal2FaceID(normalWorld);

    vec3 radiance;
    uint sample_count = 0;
    // Sample the hit location in the spatial hash
    bool found = SpatialHashGet(key, radiance, sample_count);

    if (found) {
        // Add to surfel value
        SurfelEntry surfel = s_surfel_pool.entries[gl_LaunchIDEXT.x];

        SpatialHashKey key;
        key.position = surfel.position;
        key.direction = uint8_t(surfel.direction);
        // Insert into the spatial hash where the ray was spanwed
        // Outgoing radiace at the current voxel is the incoming radiance from the other voxel times the albedo
        SpatialHashInsert(key, radiance * 0.9); // TODO: times the actual albedo
        // TODO: Optimize the Insert / Get flow so they only do the loop once.

        // TODO: Maybe add more samples to the patch: need heuristic.
    } else {
        // TODO: enqueue the patch stocastically
        // TODO: Also need to enqueue a sample of strength = 0
        // TODO: weight expotential fallout over time
    }
}
