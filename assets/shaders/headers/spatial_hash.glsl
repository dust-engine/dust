layout(constant_id = 0) const uint32_t SpatialHashCapacity = 32 * 1024 * 1024; // 512 MB

struct SpatialHashKey {
    ivec3 position;
    uint8_t direction; // [0, 6) indicating one of the six faces of the cube
};


// https://www.pcg-random.org/
uint pcg(in uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;

    return (word >> 22u) ^ word;
}

// xxhash (https://github.com/Cyan4973/xxHash)
//   From: https://www.shadertoy.com/view/Xt3cDn
uint xxhash32(in uint p)
{
    const uint PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
    const uint PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;

    uint h32 = p + PRIME32_5;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 = PRIME32_2 * (h32 ^ (h32 >> 15));
    h32 = PRIME32_3 * (h32 ^ (h32 >> 13));

    return h32 ^ (h32 >> 16);
}

uint32_t SpatialHashKeyGetFingerprint(SpatialHashKey key) {
    uint hash = xxhash32(key.position.x);
    hash = xxhash32(key.position.y + hash);
    hash = xxhash32(key.position.z + hash);
    hash = xxhash32(key.direction + hash);
    hash = max(1, hash);
    return hash;
}
uint32_t SpatialHashKeyGetLocation(SpatialHashKey key) {
    uint hash = pcg(key.position.x);
    hash = pcg(key.position.y + hash);
    hash = pcg(key.position.z + hash);
    hash = pcg(key.direction + hash);
    return hash % SpatialHashCapacity;
}




void SpatialHashInsert(SpatialHashKey key, vec3 value) {
    uint fingerprint = SpatialHashKeyGetFingerprint(key);
    uint location = SpatialHashKeyGetLocation(key);


    uint i_minFrameIndex;
    uint minFrameIndex;
    for (uint i = 0; i < 3; i++) {
        // Compare the stored fingerprint with 0.
        // If it's 0, populate with the new fingerprint.
        uint current_fingerprint = atomicCompSwap(
            spatial_hash[location + i].fingerprint, // 1
            0,
            fingerprint
        );
        uint current_frame_index = spatial_hash[location + i].last_accessed_frame;
        if (i == 0 || current_frame_index < minFrameIndex) {
            // Maintain the least recently accessed frame index.
            i_minFrameIndex = i;
            minFrameIndex = current_frame_index;
        }


        if (current_fingerprint == fingerprint || current_fingerprint == 0) {
            // Found.

            vec3 current_radiance = vec3(0.0);
            uint current_sample_count = 0;
            if (current_fingerprint == fingerprint) {
                current_sample_count = spatial_hash[location + i].sample_count;
                current_radiance = spatial_hash[location + i].radiance;
            }
            #define MAX_SAMPLE_COUNT 2048
            current_sample_count = min(current_sample_count, MAX_SAMPLE_COUNT - 1);
            uint next_sample_count = current_sample_count + 1;
            current_radiance = mix(current_radiance, value, 1.0 / float(next_sample_count));
            
            spatial_hash[location + i].radiance = current_radiance;
            spatial_hash[location + i].last_accessed_frame = uint16_t(push_constants.frame_index);
            spatial_hash[location + i].sample_count = uint16_t(next_sample_count);
            return;
        }
    }
    // Not found after 3 iterations. Evict the LRU entry.
    spatial_hash[location + i_minFrameIndex].fingerprint = fingerprint;
    spatial_hash[location + i_minFrameIndex].radiance = value;
    spatial_hash[location + i_minFrameIndex].last_accessed_frame = uint16_t(push_constants.frame_index);
    spatial_hash[location + i_minFrameIndex].sample_count = uint16_t(1);
}


// Returns: found
// out vec3: value. The outgoing radiance at the voxel.
bool SpatialHashGet(SpatialHashKey key, out vec3 value, out uint sample_count) {
    uint fingerprint = SpatialHashKeyGetFingerprint(key);
    uint location = SpatialHashKeyGetLocation(key);
    value = vec3(0.0);
    sample_count = 0;
    for (uint i = 0; i < 3; i++) {
        uint current_fingerprint = spatial_hash[location + i].fingerprint;
        if (current_fingerprint == 0) {
            return false; // Found an empty entry, so we terminate the search early.
        }

        if (current_fingerprint == fingerprint) {
            // Found.
            spatial_hash[location + i].last_accessed_frame = uint16_t(push_constants.frame_index);
            value = spatial_hash[location + i].radiance;
            sample_count = spatial_hash[location + i].sample_count;
            return true;
        }
    }
    return false;
}
