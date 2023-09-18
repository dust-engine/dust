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
layout(location = 0) rayPayloadInEXT struct RayPayload {
    vec3 illuminance;
} payload;


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



    vec3 radiance = vec3(0.0);
    bool found = SpatialHashGet(key, radiance);

    if (!found) {
        // TODO: Enqueue the patch stocastically
    } else {
        // Maybe add more samples to the patch? need heuristic.
    }

    vec4 packed = REBLUR_FrontEnd_PackRadianceAndNormHitDist(payload.illuminance + radiance, gl_HitTEXT);
    imageStore(u_illuminance, ivec2(gl_LaunchIDEXT.xy), packed);
}

// TODO: final gather and surfel should both use the corser grid.
