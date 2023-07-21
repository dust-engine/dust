
#version 460
#include "standard.glsl"

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
    vec3 sky_illuminance = arhosek_sky_radiance(normalize(gl_WorldRayDirectionEXT));

    uint rng = hash1(hash2(gl_LaunchIDEXT.xy) + pushConstants.rand);
    
    Sample initialSample;
    initialSample.sample_point = vec3(0.0);
    initialSample.sample_point_normal = vec3(0.0);
    initialSample.visible_point = gl_WorldRayOriginEXT;
    initialSample.visible_point_normal = imageLoad(u_normal, ivec2(gl_LaunchIDEXT.xy)).xyz;
    initialSample.outgoing_radiance = sky_illuminance;


    Reservoir reservoir = TemporalResampling(initialSample, rng);

    // Equation 6, where the spatial reservoir’s W weight gives all of the
    // factors other than f(y), which is evaluated as the product of the
    // BSDF, cosine factor, and the reservoir sample’s outgoing radiance

    // min(1e-8) to avoid NaN
    float W = reservoir.total_weight / max(1e-8, reservoir.sample_count * sample_target_pdf(reservoir.current_sample));
    vec3 resampled_radiance = W * reservoir.current_sample.outgoing_radiance;
    imageStore(u_illuminance, ivec2(gl_LaunchIDEXT.xy), vec4(payload.illuminance + resampled_radiance, 1.0));
}
