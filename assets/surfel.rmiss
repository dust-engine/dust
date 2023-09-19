#version 460
#include "standard.glsl"


void main() {
    vec3 sky_illuminance = arhosek_sky_radiance(normalize(gl_WorldRayDirectionEXT));

    SurfelEntry surfel = s_surfel_pool.entries[gl_LaunchIDEXT.x];

    SpatialHashKey key;
    key.position = surfel.position;
    key.direction = uint8_t(surfel.direction);
    uint fingerprint = SpatialHashKeyGetFingerprint(key);
    uint location = SpatialHashKeyGetLocation(key);


    uint i_minFrameIndex;
    uint minFrameIndex;
    for (uint i = 0; i < 3; i++) {
        uint current_fingerprint = s_spatial_hash.entries[location + i].fingerprint;
        uint current_frame_index = s_spatial_hash.entries[location + i].last_accessed_frame;
        if (i == 0 || current_frame_index < minFrameIndex) {
            i_minFrameIndex = i;
            minFrameIndex = current_frame_index;
        }


        if (current_fingerprint == fingerprint || current_fingerprint == 0) {
            // Found.
            if (current_fingerprint == 0) {
                s_spatial_hash.entries[location + i].fingerprint = fingerprint;
            }

            vec3 current_radiance = vec3(0.0);
            uint current_sample_count = 0;
            if (current_fingerprint == fingerprint) {
                current_sample_count = s_spatial_hash.entries[location + i].sample_count;
                current_radiance = s_spatial_hash.entries[location + i].radiance;
            }
            uint next_sample_count = current_sample_count + 1;
            current_radiance = current_radiance * (float(current_sample_count) / float(next_sample_count)) + sky_illuminance * (1.0 / float(next_sample_count));
            
            s_spatial_hash.entries[location + i].radiance = f16vec3(current_radiance);
            s_spatial_hash.entries[location + i].last_accessed_frame = uint16_t(pushConstants.frameIndex);
            s_spatial_hash.entries[location + i].sample_count = uint16_t(next_sample_count);
            return;
        }
    }
    // Not found after 3 iterations. Evict the LRU entry.
    s_spatial_hash.entries[location + i_minFrameIndex].fingerprint = fingerprint;
    uint current_sample_count = s_spatial_hash.entries[location + i_minFrameIndex].sample_count;
    vec3 current_radiance = s_spatial_hash.entries[location + i_minFrameIndex].radiance;
    uint next_sample_count = current_sample_count + 1;
    current_radiance = current_radiance * (float(current_sample_count) / float(next_sample_count)) + sky_illuminance * (1.0 / float(next_sample_count));
    s_spatial_hash.entries[location + i_minFrameIndex].radiance = f16vec3(current_radiance);
    s_spatial_hash.entries[location + i_minFrameIndex].last_accessed_frame = uint16_t(pushConstants.frameIndex);
    s_spatial_hash.entries[location + i_minFrameIndex].sample_count = uint16_t(next_sample_count);
}
