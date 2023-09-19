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
    uint sample_count = 0;
    bool found = SpatialHashGet(key, radiance, sample_count);
    float probability_to_schedule = 1.0 / float(sample_count + 2);
    float noise_sample = texelFetch(blue_noise[0], ivec2((gl_LaunchIDEXT.xy + uvec2(34, 21) + pushConstants.rand) % textureSize(blue_noise[0], 0)), 0).x;

    if (noise_sample > probability_to_schedule) {
        uint index = gl_LaunchIDEXT.x + gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x;
        index = index % SurfelPoolSize;

        SurfelEntry entry;
        entry.position = uvec3(round(aabbCenterWorld / 4.0));
        entry.direction = normal2FaceID(normalWorld);
        s_surfel_pool.entries[index] = entry;
    }

    vec4 packed = REBLUR_FrontEnd_PackRadianceAndNormHitDist(payload.illuminance, gl_HitTEXT);
    imageStore(u_illuminance, ivec2(gl_LaunchIDEXT.xy), packed);
}

// TODO: final gather and surfel should both use the corser grid.
