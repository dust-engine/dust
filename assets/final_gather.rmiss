
#version 460
#include "standard.glsl"

layout(location = 0) rayPayloadInEXT struct RayPayload {
    vec3 illuminance;
} payload;


void main() {
    vec3 sky_illuminance = arhosek_sky_radiance(normalize(gl_WorldRayDirectionEXT));

    vec4 packed = REBLUR_FrontEnd_PackRadianceAndNormHitDist(sky_illuminance + payload.illuminance, 0.0);
    imageStore(u_illuminance, ivec2(gl_LaunchIDEXT.xy), packed);
}
