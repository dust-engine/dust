#include "../headers/standard.glsl"
#include "../headers/layout.playout"
#include "../headers/sky.glsl"
#include "../headers/standard.glsl"
#include "../headers/surfel.glsl"
#include "../headers/spatial_hash.glsl"


void main() {
    vec3 sky_illuminance = arhosek_sky_radiance(normalize(gl_WorldRayDirectionEXT));

    SurfelEntry surfel = surfel_pool[gl_LaunchIDEXT.x];

    SpatialHashKey key;
    key.position = surfel.position;
    key.direction = uint8_t(surfel.direction);

    SpatialHashInsert(key, sky_illuminance);

    // TODO: stocastically sample the lights as well.
}
