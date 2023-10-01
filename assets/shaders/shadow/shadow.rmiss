#include "../headers/standard.glsl"
#include "../headers/sky.glsl"
#include "../headers/layout.glsl"

layout(location = 0) rayPayloadInEXT struct Payload {
    vec3 normal;
} payload;

#define MATH_PI 3.1415926

void main() {
    // On each frame, the ray hits the sun with probability (1.0 - cos(sunlight_config.solar_intensity.w))
    // But with the shadow rays, we hit the sun with probability 1.
    // So, we adjust the radiance of the sun by that factor.

    vec3 strength = arhosek_sun_radiance(normalize(gl_WorldRayDirectionEXT)) *
    (1.0 - cos(sunlight_config.solar_intensity.w));

    vec3 illuminance = vec3(strength) * dot(payload.normal, gl_WorldRayDirectionEXT);
    imageStore(u_illuminance, ivec2(gl_LaunchIDEXT.xy), vec4(illuminance, 1.0));
}
