// Converts normalized direction to the octahedral map (non-equal area, signed normalized).
//  - n: Normalized direction.
//  - returns: Position in octahedral map in [-1,1] for each component.
vec2 RTXDI_NormalizedVectorToOctahedralMapping(vec3 n)
{
    // Project the sphere onto the octahedron (|x|+|y|+|z| = 1) and then onto the xy-plane.
    vec2 p = vec2(n.x, n.y) * (1.0 / (abs(n.x) + abs(n.y) + abs(n.z)));

    // Reflect the folds of the lower hemisphere over the diagonals.
    if (n.z < 0.0) {
        p = vec2(
            (1.0 - abs(p.y)) * (p.x >= 0.0 ? 1.0 : -1.0),
            (1.0 - abs(p.x)) * (p.y >= 0.0 ? 1.0 : -1.0)
            );
    }

    return p;
}

// Converts point in the octahedral map to normalized direction (non-equal area, signed normalized).
//  - p: Position in octahedral map in [-1,1] for each component.
//  - returns: Normalized direction.
vec3 RTXDI_OctahedralMappingToNormalizedVector(vec2 p)
{
    vec3 n = vec3(p.x, p.y, 1.0 - abs(p.x) - abs(p.y));

    // Reflect the folds of the lower hemisphere over the diagonals.
    if (n.z < 0.0) {
        n.xy = vec2(
            (1.0 - abs(n.y)) * (n.x >= 0.0 ? 1.0 : -1.0),
            (1.0 - abs(n.x)) * (n.y >= 0.0 ? 1.0 : -1.0)
            );
    }

    return normalize(n);
}

// Encode a normal packed as 2x 16-bit snorms in the octahedral mapping.
uint RTXDI_EncodeNormalizedVectorToSnorm2x16(vec3 normal)
{
    vec2 octNormal = RTXDI_NormalizedVectorToOctahedralMapping(normal);
    return packSnorm2x16(octNormal);
}

// Decode a normal packed as 2x 16-bit snorms in the octahedral mapping.
vec3 RTXDI_DecodeNormalizedVectorFromSnorm2x16(uint packedNormal)
{
    vec2 octNormal = unpackSnorm2x16(packedNormal);
    return RTXDI_OctahedralMappingToNormalizedVector(octNormal);
}





// Encode an RGB color into a 32-bit LogLuv HDR format.
//
// The supported luminance range is roughly 10^-6..10^6 in 0.17% steps.
// The log-luminance is encoded with 14 bits and chroma with 9 bits each.
// This was empirically more accurate than using 8 bit chroma.
// Black (all zeros) is handled exactly.
uint RTXDI_EncodeRGBToLogLuv(vec3 color)
{
    // Convert ACEScg to XYZ.
    vec3 XYZ = ACEScg2XYZ(color);

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
vec3 RTXDI_DecodeLogLuvToRGB(uint packedColor)
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
    return max(XYZ2ACEScg(XYZ), 0.0);
}





struct UnpackedSample {
    vec3 direction;
    vec3 radiance;
};

struct Reservoir {
    float total_weight;
    UnpackedSample current_sample;
    uint sample_count;
};

void ReservoirAddSample(
    inout Reservoir reservoir,
    UnpackedSample new_sample,
    float weight, // multiple importance samplilng * target function * weight
    uint count,
    float rand // random number in [0,1)
) {
    reservoir.total_weight += weight;
    reservoir.sample_count += count;
    if (rand < weight / reservoir.total_weight) {
        reservoir.current_sample = new_sample;
    }
}

PackedReservoir ReservoirFinalize(
    Reservoir reservoir,
    float target_function // Target function for the currently stored sample
) {
    PackedReservoir packed_reservoir;
    packed_reservoir.weight = 1.0 / target_function * reservoir.total_weight;
    packed_reservoir.sample_count = uint16_t(min(reservoir.sample_count, 30));
    packed_reservoir.direction = RTXDI_EncodeNormalizedVectorToSnorm2x16(reservoir.current_sample.direction);
    packed_reservoir.radiance = RTXDI_EncodeRGBToLogLuv(reservoir.current_sample.radiance);
    return packed_reservoir;
}
Reservoir ReservoirNewEmpty() {
    Reservoir reservoir;
    reservoir.sample_count = 0;
    reservoir.total_weight = 0;
    reservoir.current_sample.radiance = vec3(0.0);
    reservoir.current_sample.direction = vec3(0.0);
    return reservoir;
}


Reservoir ReservoirInit(
    PackedReservoir packed_reservoir,
    float target_function, // Target function for the currently stored sample
    float mis_weight
) {
    Reservoir reservoir;
    reservoir.sample_count = packed_reservoir.sample_count;
    reservoir.total_weight = packed_reservoir.weight * target_function * mis_weight;
    reservoir.current_sample.radiance = RTXDI_DecodeLogLuvToRGB(packed_reservoir.radiance);
    reservoir.current_sample.direction = RTXDI_DecodeNormalizedVectorFromSnorm2x16(packed_reservoir.direction);
    return reservoir;
}
