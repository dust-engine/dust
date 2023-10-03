#include "../headers/standard.glsl"
#include "../headers/layout.playout"
#include "../headers/sky.glsl"
#include "../headers/nrd.glsl"

layout(location = 0) rayPayloadInEXT struct RayPayload {
    vec3 illuminance;
} payload;


void main() {
    vec3 sky_illuminance = vec3(0.0);

    #ifdef CONTRIBUTION_SECONDARY_SKYLIGHT
    sky_illuminance += arhosek_sky_radiance(normalize(gl_WorldRayDirectionEXT));
    #endif
    #ifdef CONTRIBUTION_DIRECT
    sky_illuminance += payload.illuminance;
    #endif
    vec4 packed = REBLUR_FrontEnd_PackRadianceAndNormHitDist(sky_illuminance , 0.0);

    #ifndef DEBUG_VISUALIZE_SPATIAL_HASH
    imageStore(img_illuminance, ivec2(gl_LaunchIDEXT.xy), packed);
    #endif
}
