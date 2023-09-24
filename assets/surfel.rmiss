#version 460
#include "standard.glsl"


void main() {
    vec3 sky_illuminance = arhosek_sky_radiance(normalize(gl_WorldRayDirectionEXT));

    SurfelEntry surfel = s_surfel_pool.entries[gl_LaunchIDEXT.x];

    SpatialHashKey key;
    key.position = surfel.position;
    key.direction = uint8_t(surfel.direction);

    SpatialHashInsert(key, sky_illuminance);

    // TODO: stocastically sample the lights as well.
}
