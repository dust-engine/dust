layout(constant_id = 0) const uint32_t SpatialHashCapacity = 32 * 1024 * 1024; // 512 MB


vec3 XYZ2ACEScg2(vec3 srgb) {
    mat3 transform = mat3(
        1.6410228, -0.66366285, 0.011721907,
        -0.32480323, 1.6153315, -0.0082844375,
        -0.23642465, 0.016756356, 0.9883947
    );
    return transform * srgb;
}
vec3 ACEScg2XYZ2(vec3 srgb) {
    mat3 transform = mat3(
        0.66245437, 0.2722288, -0.0055746622,
        0.13400422, 0.6740818, 0.00406073,
        0.15618773, 0.05368953, 1.0103393
    );
    return transform * srgb;
}


// Encode an RGB color into a 32-bit LogLuv HDR format.
//
// The supported luminance range is roughly 10^-6..10^6 in 0.17% steps.
// The log-luminance is encoded with 14 bits and chroma with 9 bits each.
// This was empirically more accurate than using 8 bit chroma.
// Black (all zeros) is handled exactly.
uint EncodeRGBToLogLuv(vec3 color)
{
    // Convert ACEScg to XYZ.
    vec3 XYZ = ACEScg2XYZ2(color);

    // Encode log2(Y) over the range [-20,20) in 14 bits (no sign bit).
    // TODO: Fast path that uses the bits from the fp32 representation directly.
    float logY = 409.6 * (log2(XYZ.y) + 20.0); // -inf if Y==0
    uint Le = uint(clamp(logY, 0.0, 16383.0));

    // Early out if zero luminance to avoid NaN in chroma computation.
    // Note Le==0 if Y < 9.55e-7. We'll decode that as exactly zero.
    if (Le == 0) return 0;

    // Compute chroma (u,v) values by:
    //  x = X / (X + Y + Z)
    //  y = Y / (X + Y + Z)
    //  u = 4x / (-2x + 12y + 3)
    //  v = 9y / (-2x + 12y + 3)
    //
    // These expressions can be refactored to avoid a division by:
    //  u = 4X / (-2X + 12Y + 3(X + Y + Z))
    //  v = 9Y / (-2X + 12Y + 3(X + Y + Z))
    //
    float invDenom = 1.0 / (-2.0 * XYZ.x + 12.0 * XYZ.y + 3.0 * (XYZ.x + XYZ.y + XYZ.z));
    vec2 uv = vec2(4.0, 9.0) * XYZ.xy * invDenom;

    // Encode chroma (u,v) in 9 bits each.
    // The gamut of perceivable uv values is roughly [0,0.62], so scale by 820 to get 9-bit values.
    uvec2 uve = uvec2(clamp(820.0 * uv, 0.0, 511.0));

    return (Le << 18) | (uve.x << 9) | uve.y;
}

// Decode an RGB color stored in a 32-bit LogLuv HDR format.
//    See RTXDI_EncodeRGBToLogLuv() for details.
vec3 DecodeLogLuvToRGB(uint packedColor)
{
    // Decode luminance Y from encoded log-luminance.
    uint Le = packedColor >> 18;
    if (Le == 0) return vec3(0, 0, 0);

    float logY = (float(Le) + 0.5) / 409.6 - 20.0;
    float Y = pow(2.0, logY);

    // Decode normalized chromaticity xy from encoded chroma (u,v).
    //
    //  x = 9u / (6u - 16v + 12)
    //  y = 4v / (6u - 16v + 12)
    //
    uvec2 uve = uvec2(packedColor >> 9, packedColor) & 0x1ff;
    vec2 uv = (vec2(uve)+0.5) / 820.0;

    float invDenom = 1.0 / (6.0 * uv.x - 16.0 * uv.y + 12.0);
    vec2 xy = vec2(9.0, 4.0) * uv * invDenom;

    // Convert chromaticity to XYZ and back to RGB.
    //  X = Y / y * x
    //  Z = Y / y * (1 - x - y)
    //
    float s = Y / xy.y;
    vec3 XYZ = vec3(s * xy.x, Y, s * (1.f - xy.x - xy.y));

    // Convert back to RGB and clamp to avoid out-of-gamut colors.
    return max(XYZ2ACEScg2(XYZ), 0.0);
}




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
                current_radiance = DecodeLogLuvToRGB(spatial_hash[location + i].radiance);
            }
            #define MAX_SAMPLE_COUNT 404
            current_sample_count = min(current_sample_count, MAX_SAMPLE_COUNT - 1);
            uint next_sample_count = current_sample_count + 1;
            current_radiance = mix(current_radiance, value, 1.0 / float(next_sample_count));
            
            spatial_hash[location + i].radiance = EncodeRGBToLogLuv(current_radiance);
            spatial_hash[location + i].last_accessed_frame = uint16_t(push_constants.frame_index);
            spatial_hash[location + i].sample_count = uint16_t(next_sample_count);
            return;
        }
    }
    // Not found after 3 iterations. Evict the LRU entry.
    spatial_hash[location + i_minFrameIndex].fingerprint = fingerprint;
    spatial_hash[location + i_minFrameIndex].radiance = EncodeRGBToLogLuv(value);
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
            value = DecodeLogLuvToRGB(spatial_hash[location + i].radiance);
            sample_count = spatial_hash[location + i].sample_count;
            return true;
        }
    }
    return false;
}
