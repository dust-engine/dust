// Radiance cascades: from the rays' deposits to a cone of incoming radiance per
// direction of cascade 0, and from those to irradiance per normal (steps 5 and 6
// in rc_common.slang).

import pbr_common;
import rc_common;

[vk::push_constant] uniform RcPush pc;

// Trilinear weight of a probe's neighbor in the next cascade, by how many axes it
// is off the parent: a probe is 1/4 of the way from its parent to them, so each
// such axis is worth a third.
float neighborWeight(uint axesOff) {
    switch (axesOff) {
    case 0:
        return 27.0 / 64.0;
    case 1:
        return 9.0 / 64.0;
    case 2:
        return 3.0 / 64.0;
    default:
        return 1.0 / 64.0;
    }
}

// I_{n+1} at a probe of cascade n, averaged over the four directions merging into
// a direction of cascade n (the sum in Eq. 7). Sparse trilinear interpolation:
// only probes that exist and have a value for this direction count (§4). .a is 0
// if none do.
float4 interpolateUpper(uint cascade, RcCascade c, uint probe, uint direction) {
    RcCascade up = params.rc.cascades[cascade + 1u];
    uint directions = rcDirectionCount(cascade);
    float3 sum = float3(0.0, 0.0, 0.0);
    float weight = 0.0;
    for (uint k = 0; k < 8; k++) {
        uint neighbor = params.rc.neighbors[c.neighborOffset + probe * 8u + k];
        if (neighbor == RC_INVALID) {
            continue;
        }
        float4 value = float4(params.rc.merged[up.mergedOffset + neighbor * directions + direction]);
        if (value.a > 0.0) {
            float w = neighborWeight(countbits(k));
            sum += w * value.rgb;
            weight += w;
        }
    }
    return weight > 0.0 ? float4(sum / weight, 1.0) : float4(0.0, 0.0, 0.0, 0.0);
}

// I_n of one direction of one probe (Eq. 7): this frame's rays blended with the
// history (§5.2), and whatever gets through them coming from the next cascade. A
// direction no ray has had has no value (.a is 0): it probably points into a wall,
// and what the next cascade has for it, from probes elsewhere, doesn't apply.
float4 resolve(uint cascade, RcCascade c, uint probe, uint prevIndex, uint direction) {
    uint directions = rcDirectionCount(cascade);
    uint index = c.dataOffset + probe * directions + direction;
    RcAccum deposited = params.rc.accum[index];
    uint rays = deposited.counts & 0xFFFFu;
    float4 previous = float4(0.0, 0.0, 0.0, -1.0);
    if (prevIndex != RC_INVALID) {
        previous = float4(params.rc.prevHistory[c.dataOffset + prevIndex * directions + direction]);
    }
    float3 J = previous.rgb;
    float transmittance = previous.a;
    if (rays > 0) {
        RcAccum empty;
        empty.r = 0.0;
        empty.g = 0.0;
        empty.b = 0.0;
        empty.counts = 0u;
        params.rc.accum[index] = empty;
        J = float3(deposited.r, deposited.g, deposited.b) / float(rays);
        transmittance = float(deposited.counts >> 16) / float(rays);
        if (previous.a >= 0.0) {
            J = lerp(previous.rgb, J, params.rc.temporalAlpha);
            transmittance = lerp(previous.a, transmittance, params.rc.temporalAlpha);
        }
    } else if (previous.a < 0.0) {
        params.rc.history[index] = float16_t4(previous);
        return float4(0.0, 0.0, 0.0, 0.0);
    }
    params.rc.history[index] = float16_t4(float4(J, transmittance));

    // J is the sum over the rays stopped by this interval, over all the rays.
    float known = 1.0 - transmittance;
    if (transmittance > 0.0 && cascade + 1u < RC_CASCADES) {
        float4 upper = interpolateUpper(cascade, c, probe, direction);
        if (upper.a > 0.0) {
            J += transmittance * upper.rgb;
            known = 1.0;
        }
    }
    return known > 0.0 ? float4(J / known, 1.0) : float4(0.0, 0.0, 0.0, 0.0);
}

// Each cascade above 0 is only ever read by the one below, so it stores what that
// reads: its values pre-averaged per direction of that cascade, leaving out those
// with no value (§5, §6).
[shader("compute")]
[numthreads(32, 1, 1)]
void rcMerge(uint3 threadId: SV_DispatchThreadID) {
    uint cascade = pc.cascade;
    RcCascade c = params.rc.cascades[cascade];
    uint outputs = rcDirectionCount(cascade == 0 ? 0u : cascade - 1u);
    uint probe = threadId.x / outputs;
    uint direction = threadId.x % outputs;
    uint prevIndex = params.rc.probes[c.probeOffset + probe].prevIndex;
    float3 sum = float3(0.0, 0.0, 0.0);
    float known = 0.0;
    if (cascade == 0) {
        float4 value = resolve(cascade, c, probe, prevIndex, direction);
        sum = value.rgb;
        known = value.a;
    } else {
        for (uint child = 0; child < 4; child++) {
            float4 value = resolve(cascade, c, probe, prevIndex, direction * 4u + child);
            sum += value.rgb;
            known += value.a;
        }
    }
    float4 average = known > 0.0 ? float4(sum / known, 1.0) : float4(0.0, 0.0, 0.0, 0.0);
    params.rc.merged[c.mergedOffset + probe * outputs + direction] = float16_t4(average);
}

// A probe's tile: the rendering equation evaluated per normal over the 32 cones,
// for a 6x6 octahedral map, with a border copied from across the map's edges so
// that bilinear sampling doesn't see the next tile (§6). A texel holds the mean
// radiance weighted by cosine, times the known fraction of its hemisphere (which
// is its alpha): a probe on the far side of a wall knows nothing of the near side,
// and sampling skips it instead of darkening.
//
// One group per probe and one thread per texel, with no local arrays: on Apple
// GPUs those live in memory, and an array of the cones here took far longer than
// the arithmetic.
[shader("compute")]
[numthreads(RC_TILE_INNER * RC_TILE_INNER, 1, 1)]
void rcIrradiance(uint3 groupId: SV_GroupID, uint texelIndex: SV_GroupIndex) {
    uint probe = groupId.x;
    int last = int(RC_TILE_INNER) - 1;
    int2 texel = int2(texelIndex % RC_TILE_INNER, texelIndex / RC_TILE_INNER);
    float3 normal = rcOctDecode((float2(texel) + 0.5) / float(RC_TILE_INNER));

    // The directions of cascade 0 are 4 rings of 8, in the order of their indices:
    // rcDecodeDirection at cell centers. Each is the last rotated by 45 degrees.
    uint first = params.rc.cascades[0].mergedOffset + probe * rcDirectionCount(0);
    float3 sum = float3(0.0, 0.0, 0.0);
    float known = 0.0;
    float total = 0.0;
    for (uint ring = 0; ring < 4; ring++) {
        float z = (float(ring) + 0.5) * 0.5 - 1.0;
        float r = sqrt(1.0 - z * z);
        // cos and sin of 22.5 degrees.
        float2 around = float2(0.92387953, 0.38268343);
        for (uint step = 0; step < 8; step++) {
            float cosine = max(dot(normal, float3(r * around, z)), 0.0);
            total += cosine;
            if (cosine > 0.0) {
                float4 cone = float4(params.rc.merged[first + ring * 8u + step]);
                known += cosine * cone.a;
                sum += cosine * cone.rgb;
            }
            around = float2(around.x - around.y, around.x + around.y) * 0.70710678;
        }
    }
    float4 value = float4(sum, known) / total;

    uint tilesPerRow = params.rc.own.tilesPerRow;
    int2 tile = int2(probe % tilesPerRow, probe / tilesPerRow) * int(RC_TILE_SIZE) + int2(1, 1);
    params.rcIrradianceOutput[uint2(tile + texel)] = value;
    // Crossing an edge of the map lands on that edge, mirrored about its middle,
    // so this texel is also the border just beyond the mirrored spot. Each corner
    // of the map is the opposite corner of its tile, having crossed two edges.
    if (texel.x == 0) {
        params.rcIrradianceOutput[uint2(tile + int2(-1, last - texel.y))] = value;
    }
    if (texel.x == last) {
        params.rcIrradianceOutput[uint2(tile + int2(last + 1, last - texel.y))] = value;
    }
    if (texel.y == 0) {
        params.rcIrradianceOutput[uint2(tile + int2(last - texel.x, -1))] = value;
    }
    if (texel.y == last) {
        params.rcIrradianceOutput[uint2(tile + int2(last - texel.x, last + 1))] = value;
    }
    if ((texel.x == 0 || texel.x == last) && (texel.y == 0 || texel.y == last)) {
        int2 outward = (texel == 0) * -1 + (texel == last) * 1;
        params.rcIrradianceOutput[uint2(tile + int2(last, last) - texel - outward)] = value;
    }
}
