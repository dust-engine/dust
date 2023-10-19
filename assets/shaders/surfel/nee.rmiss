#include "../headers/standard.glsl"
#include "../headers/layout.playout"
#include "../headers/color.glsl"
#include "../headers/sky.glsl"
#include "../headers/standard.glsl"
#include "../headers/surfel.glsl"
#include "../headers/spatial_hash.glsl"
#include "../headers/normal.glsl"

layout(location = 0) rayPayloadInEXT struct RayPayload {
    vec3 radiance;
} payload;


void main() {
    SurfelEntry surfel = surfel_pool[gl_LaunchIDEXT.x];

    SpatialHashKey key;
    key.position = ivec3((surfel.position / 4.0));
    key.direction = uint8_t(surfel.direction);

    vec3 normal = faceId2Normal(uint8_t(surfel.direction)); // world space
    vec3 strength = arhosek_sun_radiance(normalize(gl_WorldRayDirectionEXT)) * (1.0 - cos(sunlight_config.solar_intensity.w));
    vec3 illuminance = vec3(strength) * dot(normal, gl_WorldRayDirectionEXT);

    payload.radiance += illuminance;
}
