// Radiance cascades: from the rays' deposits to a cone of incoming radiance per
// direction of cascade 0, and from those to irradiance per normal (steps 5 and 6
// in rc_common.slang).

import pbr_common;
import rc_common;

[vk::push_constant] uniform RcPush pc;

// Trilinear weights of a probe's neighbors in the next cascade, by how many axes
// a neighbor is off the parent: a probe is 1/4 of the way from its parent to them.
static const float RC_NEIGHBOR_WEIGHTS[4] = { 27.0 / 64.0, 9.0 / 64.0, 3.0 / 64.0, 1.0 / 64.0 };

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
            float w = RC_NEIGHBOR_WEIGHTS[countbits(k)];
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

// The directions of cascade 0, by index: the centers of rcDecodeDirection's cells.
static const float3 RC_C0_DIRECTIONS[32] = {
    float3(0.6110889, 0.2531213, -0.7500000), float3(0.2531213, 0.6110889, -0.7500000),
    float3(-0.2531213, 0.6110889, -0.7500000), float3(-0.6110889, 0.2531213, -0.7500000),
    float3(-0.6110889, -0.2531213, -0.7500000), float3(-0.2531213, -0.6110889, -0.7500000),
    float3(0.2531213, -0.6110889, -0.7500000), float3(0.6110889, -0.2531213, -0.7500000),
    float3(0.8945425, 0.3705316, -0.2500000), float3(0.3705316, 0.8945425, -0.2500000),
    float3(-0.3705316, 0.8945425, -0.2500000), float3(-0.8945425, 0.3705316, -0.2500000),
    float3(-0.8945425, -0.3705316, -0.2500000), float3(-0.3705316, -0.8945425, -0.2500000),
    float3(0.3705316, -0.8945425, -0.2500000), float3(0.8945425, -0.3705316, -0.2500000),
    float3(0.8945425, 0.3705316, 0.2500000), float3(0.3705316, 0.8945425, 0.2500000),
    float3(-0.3705316, 0.8945425, 0.2500000), float3(-0.8945425, 0.3705316, 0.2500000),
    float3(-0.8945425, -0.3705316, 0.2500000), float3(-0.3705316, -0.8945425, 0.2500000),
    float3(0.3705316, -0.8945425, 0.2500000), float3(0.8945425, -0.3705316, 0.2500000),
    float3(0.6110889, 0.2531213, 0.7500000), float3(0.2531213, 0.6110889, 0.7500000),
    float3(-0.2531213, 0.6110889, 0.7500000), float3(-0.6110889, 0.2531213, 0.7500000),
    float3(-0.6110889, -0.2531213, 0.7500000), float3(-0.2531213, -0.6110889, 0.7500000),
    float3(0.2531213, -0.6110889, 0.7500000), float3(0.6110889, -0.2531213, 0.7500000),
};

// A probe's tile: the rendering equation evaluated per normal over the 32 cones,
// for a 6x6 octahedral map, then a border copied in from across the map's edges so
// that bilinear sampling doesn't see the next tile (§6). A texel holds the mean
// radiance weighted by cosine, times the known fraction of its hemisphere (which
// is its alpha): a probe on the far side of a wall knows nothing of the near side,
// and sampling skips it instead of darkening.
[shader("compute")]
[numthreads(64, 1, 1)]
void rcIrradiance(uint3 threadId: SV_DispatchThreadID) {
    uint probe = threadId.x;
    if (probe >= params.rc.counters[RC_COUNTER_PROBES]) {
        return;
    }
    uint first = params.rc.cascades[0].mergedOffset + probe * rcDirectionCount(0);
    float4 cones[32];
    for (uint i = 0; i < 32; i++) {
        cones[i] = float4(params.rc.merged[first + i]);
    }
    float4 texels[RC_TILE_INNER * RC_TILE_INNER];
    for (uint y = 0; y < RC_TILE_INNER; y++) {
        for (uint x = 0; x < RC_TILE_INNER; x++) {
            float3 normal = rcOctDecode((float2(x, y) + 0.5) / float(RC_TILE_INNER));
            float3 sum = float3(0.0, 0.0, 0.0);
            float known = 0.0;
            float total = 0.0;
            for (uint i = 0; i < 32; i++) {
                float cosine = max(dot(normal, RC_C0_DIRECTIONS[i]), 0.0);
                total += cosine;
                known += cosine * cones[i].a;
                sum += cosine * cones[i].rgb;
            }
            texels[y * RC_TILE_INNER + x] = float4(sum, known) / total;
        }
    }
    uint tilesPerRow = params.rc.own.tilesPerRow;
    int2 tile = int2(probe % tilesPerRow, probe / tilesPerRow) * int(RC_TILE_SIZE);
    int last = int(RC_TILE_INNER) - 1;
    for (int y = -1; y <= last + 1; y++) {
        for (int x = -1; x <= last + 1; x++) {
            // Crossing an edge of the map lands on that edge, mirrored about its
            // middle. Corners cross two.
            int2 inner = int2(x, y);
            if (inner.x < 0 || inner.x > last) {
                inner = int2(clamp(inner.x, 0, last), last - inner.y);
            }
            if (inner.y < 0 || inner.y > last) {
                inner = int2(last - inner.x, clamp(inner.y, 0, last));
            }
            params.rcIrradianceOutput[uint2(tile + int2(x + 1, y + 1))] = texels[inner.y * int(RC_TILE_INNER) + inner.x];
        }
    }
}
