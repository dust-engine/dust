
#version 460
#include "standard.glsl"

layout(location = 0) rayPayloadInEXT struct RayPayload {
    vec3 illuminance;
} payload;



void main() {
    vec3 sky_illuminance = arhosek_sky_radiance(normalize(gl_WorldRayDirectionEXT));
    // TODO: calculate ambient light, add into main texture. We assume that the ambient light is 0.1.
    imageStore(u_illuminance, ivec2(gl_LaunchIDEXT.xy), vec4(payload.illuminance + sky_illuminance, 1.0));
}
