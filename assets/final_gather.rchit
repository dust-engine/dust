#version 460
#include "standard.glsl"

layout(shaderRecordEXT) buffer Sbt {
    GeometryInfo geometryInfo;
    MaterialInfo materialInfo;
    PaletteInfo paletteInfo;
    IrradianceCache irradianceCache;
} sbt;

hitAttributeEXT HitAttribute {
    uint8_t voxelId;
} hitAttributes;
layout(location = 0) rayPayloadInEXT struct RayPayload {
    vec3 illuminance;
} payload;

float sample_source_pdf(Sample s) {
    return max(0.0, dot(s.visible_point_normal, gl_WorldRayDirectionEXT));
}
float sample_target_pdf(Sample s) {
    return s.outgoing_radiance.y;
}

Reservoir TemporalResampling(Sample initial_sample, inout uint rng) {
	vec2 uv = (gl_LaunchIDEXT.xy + vec2(0.5)) / gl_LaunchSizeEXT.xy;
	vec2 motion = imageLoad(u_motion, ivec2(gl_LaunchIDEXT.xy)).xy;
	uv += motion;
    
    Reservoir reservoir;
	if ((uv.x < 0.0) || (uv.x > 1.0) || (uv.y < 0.0) || (uv.y > 1.0)) {
        // disocclusion
        reservoir.sample_count = 0;
        reservoir.total_weight = 0;
        reservoir.current_sample = initial_sample;
	} else {
        uvec2 pos = uvec2(uv * vec2(gl_LaunchSizeEXT.xy));
        reservoir = s_reservoirs_prev.reservoirs[pos.x * imageSize(u_illuminance).y + pos.y];
    }
    if (reservoir.sample_count > 20) {
        reservoir.total_weight *= 20 / float(reservoir.sample_count);
        reservoir.sample_count = 20;
    }
    float w = sample_target_pdf(initial_sample) / sample_source_pdf(initial_sample);
    ReservoirUpdate(reservoir, initial_sample, w, rng);

    // TODO: now, store Rreservoir.
    s_reservoirs.reservoirs[gl_LaunchIDEXT.x * imageSize(u_illuminance).y + gl_LaunchIDEXT.y] = reservoir;
    return reservoir;
}

void main() {
    
    uint rng = hash1(hash2(gl_LaunchIDEXT.xy) + pushConstants.rand);


    Block block = sbt.geometryInfo.blocks[gl_PrimitiveID];

    vec3 hitPointObject = gl_HitTEXT * gl_ObjectRayDirectionEXT + gl_ObjectRayOriginEXT;
    vec3 offsetInBox = vec3(hitAttributes.voxelId >> 4, (hitAttributes.voxelId >> 2) & 3, hitAttributes.voxelId & 3);
    vec3 boxCenterObject = block.position.xyz + offsetInBox + vec3(0.5);
    vec3 normalObject = CubedNormalize(hitPointObject - boxCenterObject);

    int8_t faceId = int8_t(normalObject.x) * int8_t(3) + int8_t(normalObject.y) * int8_t(2) + int8_t(normalObject.z);
    uint8_t faceIdU = uint8_t(min((faceId > 0 ? (faceId-1) : (6 + faceId)), 5));

    uint16_t mask = sbt.irradianceCache.entries[gl_PrimitiveID].faces[faceIdU].mask;
    if (mask == 0) {
        return;
    }
    uint16_t lastAccessedFrameIndex = sbt.irradianceCache.entries[gl_PrimitiveID].lastAccessedFrameIndex[faceIdU];

    // irradiance, pre multiplied with albedo.
    vec3 irradiance = vec3(sbt.irradianceCache.entries[gl_PrimitiveID].faces[faceIdU].irradiance) * pow(RETENTION_FACTOR, uint16_t(pushConstants.frameIndex) - lastAccessedFrameIndex);

    float scaling_factors =
    // Divide by the activated surface area of the voxel
    (1.0 / float(bitCount(uint(mask)))) *
    // Correction based on the retention factor.
    /// \sigma a^n from 0 to inf is 1 / (1 - a).
    (1.0 - RETENTION_FACTOR)
    ;
    vec3 radiance = irradiance * scaling_factors;

    Sample initialSample;
    initialSample.sample_point = gl_HitTEXT * gl_WorldRayDirectionEXT + gl_WorldRayOriginEXT;
    initialSample.sample_point_normal = gl_ObjectToWorldEXT * vec4(normalObject, 0.0);
    initialSample.visible_point = gl_WorldRayOriginEXT;
    initialSample.visible_point_normal = imageLoad(u_normal, ivec2(gl_LaunchIDEXT.xy)).xyz;
    initialSample.outgoing_radiance = radiance;



    Reservoir reservoir = TemporalResampling(initialSample, rng);

    // Equation 6, where the spatial reservoir’s W weight gives all of the
    // factors other than f(y), which is evaluated as the product of the
    // BSDF, cosine factor, and the reservoir sample’s outgoing radiance

    // min(1e-8) to avoid NaN
    float W = reservoir.total_weight / max(1e-8, reservoir.sample_count * sample_target_pdf(reservoir.current_sample));
    vec3 resampled_radiance = W * reservoir.current_sample.outgoing_radiance;
    //imageStore(u_illuminance, ivec2(gl_LaunchIDEXT.xy), vec4(payload.illuminance + resampled_radiance, 1.0));
}
