#include "../headers/standard.glsl"
#include "../headers/layout.playout"
#include "../headers/sky.glsl"
#include "../headers/nrd.glsl"

layout(location = 0) rayPayloadInEXT struct RayPayload {
    vec3 illuminance;
} payload;


void main() {
    vec3 sky_illuminance = arhosek_sky_radiance(normalize(gl_WorldRayDirectionEXT));

    vec4 packed = REBLUR_FrontEnd_PackRadianceAndNormHitDist(sky_illuminance , 0.0);
    imageStore(img_illuminance, ivec2(gl_LaunchIDEXT.xy), packed);
}
