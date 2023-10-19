#include "../headers/standard.glsl"
#include "../headers/layout.playout"
#include "../headers/color.glsl"
#include "../headers/sky.glsl"
#include "../headers/nrd.glsl"

layout(location = 0) rayPayloadInEXT struct RayPayload {
    vec3 illuminance;
} payload;

void main() {
    float unused;
    vec3 normalWorld = NRD_FrontEnd_UnpackNormalAndRoughness(imageLoad(img_normal, ivec2(gl_LaunchIDEXT.xy)), unused).xyz;
    // On each frame, the ray hits the sun with probability (1.0 - cos(sunlight_config.solar_intensity.w))
    // But with the shadow rays, we hit the sun with probability 1.
    // So, we adjust the radiance of the sun by that factor.

    vec3 strength = arhosek_sun_radiance(normalize(gl_WorldRayDirectionEXT)) *
    (1.0 - cos(sunlight_config.solar_intensity.w));

    payload.illuminance += strength * dot(normalWorld, gl_WorldRayDirectionEXT);
}

