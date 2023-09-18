#version 460
#include "standard.glsl"

struct SurfelEntry { 
    uvec3 position;
    uint32_t direction; // [0, 6) indicating one of the six faces of the cube
};

layout(set = 0, binding = 13) buffer SurfelPool {
    SurfelEntry entries[];
} s_surfel_pool;

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
    key.position = uvec3(round(aabbCenterWorld / 4.0));
    key.direction = normal2FaceID(normalWorld);

    vec3 radiance;
    bool found = SpatialHashGet(key, radiance);

    if (found) {
        // Add to surfel value
        SurfelEntry surfel = s_surfel_pool.entries[gl_LaunchIDEXT.x];

        SpatialHashKey key;
        key.position = surfel.position;
        key.direction = uint8_t(surfel.direction);
        SpatialHashInsert(key, radiance);

        // TODO: Maybe add more samples to the patch: need heuristic.
    } else {
        // TODO: enqueue the patch stocastically
    }
}
