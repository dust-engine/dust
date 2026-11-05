// Split Radiance Cascades (Freeman, Sannikov and Pitt-Francis): diffuse global
// illumination from sparse world-space radiance probes. Section numbers below
// refer to the paper.
//
// A cache has four cascades of probes. Going up a cascade doubles the spacing,
// quadruples the directions per probe and quadruples the length of the interval
// each direction covers (§3.2). Probes exist only where something visible needs
// them: a hash table per cascade holds the cells that surfaces fall into (§4),
// and doubling the probes' scale with distance from the camera (§4.1, LODs) keeps
// their count, and their size on screen, about constant.
//
// Rays don't leave from probes, whose positions are often inside walls. They
// leave from the visible surfaces, and each is split into the intervals it covers
// (§5): for each cascade below the one its hit falls in, the probe around the
// ray's origin records that the direction is clear, and for that cascade, the
// radiance it hit. The cascades then merge back to front into a cone per
// direction of cascade 0, which is turned into irradiance per normal (§6).
//
// Each frame (rc_build.slang, rc_trace.slang, rc_merge.slang, rc_shade.slang):
//   1. Hash the cell of every source point (a pixel or a ray hit).
//   2. Give each cell a probe, and its parent cell in the next cascade likewise.
//   3. Hand each probe a contiguous range of the R2 sequence, top cascade first,
//      so the rays of each cascade's probe are evenly spread too (§5.1).
//   4. Trace, depositing radiance and transmittance with atomic adds.
//   5. Blend with last frame (§5.2) and merge, top cascade first.
//   6. Fill each cascade 0 probe's octahedral irradiance tile, and shade.
//
// Multiple bounces (§6): a secondary cache takes the points last frame's rays hit
// as its source points. Hits of either cache add its irradiance, so light keeps
// bouncing over the frames.
//
// Hash keys are 32 bits, since 64-bit atomics can't be assumed: 5 for the LOD and
// 9 per axis, wrapped. That is exact because a LOD's cells within reach of the
// camera span under 512 per axis. `RadianceCascadesConfig` keeps it so.
module rc_common;
import pbr_common;
import random;

public static const uint RC_CASCADES = 4;
public static const uint RC_INVALID = 0xFFFFFFFFu;
// Linear probing gives up after this many slots, dropping the insertion.
static const uint RC_MAX_PROBE_STEPS = 32;
static const uint RC_MAX_LOD = 29;
// Texels per axis of a probe's irradiance tile, without and with its border.
public static const uint RC_TILE_INNER = 6;
public static const uint RC_TILE_SIZE = 8;

// The push constants of the compute passes, after the ray generation stage's
// range (pbr.playout.ron).
public struct RcPush {
    [vk::offset(4)] public uint cascade;
}

// Directions per probe of a cascade: the paper's 2 * Θ_n^2 with Θ_0 = 4.
public uint rcDirectionCount(uint cascade) {
    return 32u << (2u * cascade);
}

// Whether the G-buffer holds a surface at `pixel`: sky stores depth 0 and UI 1.
public bool rcIsSurfacePixel(uint2 pixel) {
    float z = params.gbuffer.depthTexture[pixel];
    return z > 0.0 && z < 1.0;
}

// ── Levels of detail (§4.1) ──────────────────────────────────────────────

public struct RcLod {
    public uint level;
    // Weight of `level + 1` instead: LODs overlap over the last 10% of a LOD's
    // range, where shading fades from one to the other.
    public float blend;
}

public RcLod rcLodOf(float3 position) {
    float3 offset = abs(position - params.rc.cameraPosition);
    float distance = max(offset.x, max(offset.y, offset.z)) / params.rc.lodDistance;
    uint level = distance < 1.0 ? 0u : min(uint(floor(log2(distance))) + 1u, RC_MAX_LOD);
    RcLod lod;
    lod.level = level;
    lod.blend = saturate((distance * exp2(-float(level)) - 0.9) * 10.0);
    return lod;
}

public float rcSpacing(uint lod, uint cascade) {
    return params.rc.spacing * exp2(float(lod + cascade));
}

// Where a surface point counts as being for the probe grid: just off the surface,
// so that voxel faces, which are often exactly on a cell boundary, consistently
// land in the cell they face.
public float3 rcGridPoint(float3 position, float3 normal, uint lod) {
    return position + normal * 0.05 * rcSpacing(lod, 0);
}

// ── Hash tables (§6) ─────────────────────────────────────────────────────

public int3 rcCellOf(float3 position, uint lod, uint cascade) {
    return int3(floor(position / rcSpacing(lod, cascade)));
}

// Never 0, which marks an empty slot.
public uint rcKey(uint lod, int3 cell) {
    uint3 wrapped = uint3(cell) & 511u;
    return ((lod + 1u) << 27) | (wrapped.x << 18) | (wrapped.y << 9) | wrapped.z;
}

// The cell behind a key: the one with the wrapped coordinates nearest the camera.
public int3 rcCellOfKey(uint key, uint cascade, out uint lod) {
    lod = (key >> 27) - 1u;
    int3 wrapped = int3((key >> 18) & 511u, (key >> 9) & 511u, key & 511u);
    int3 center = rcCellOf(params.rc.cameraPosition, lod, cascade);
    int3 difference = (wrapped - center) & 511;
    return center + (difference ^ 256) - 256;
}

uint rcHash(uint key) {
    key ^= key >> 16;
    key *= 0x85EBCA6Bu;
    key ^= key >> 13;
    key *= 0xC2B2AE35u;
    return key ^ (key >> 16);
}

// The hash slot of `key` in a cascade's table, adding it if it's new; RC_INVALID
// if the table is too full.
public uint rcInsert(uint cascade, uint key) {
    RcCascade c = params.rc.cascades[cascade];
    uint hash = rcHash(key);
    for (uint i = 0; i < RC_MAX_PROBE_STEPS; i++) {
        uint slot = (hash + i) & c.tableMask;
        uint current = params.rc.keys[c.tableOffset + slot];
        if (current == 0) {
            InterlockedCompareExchange(params.rc.keys[c.tableOffset + slot], 0u, key, current);
            if (current == 0) {
                return slot;
            }
        }
        if (current == key) {
            return slot;
        }
    }
    return RC_INVALID;
}

// The index of the probe of `key` in a table whose probes have been allocated;
// RC_INVALID if there is none.
public uint rcFindProbe(uint* keys, uint* slots, uint tableOffset, uint tableMask, uint key) {
    uint hash = rcHash(key);
    for (uint i = 0; i < RC_MAX_PROBE_STEPS; i++) {
        uint slot = tableOffset + ((hash + i) & tableMask);
        uint current = keys[slot];
        if (current == key) {
            return slots[slot];
        }
        if (current == 0) {
            break;
        }
    }
    return RC_INVALID;
}

public uint rcFindProbe(uint cascade, uint key) {
    RcCascade c = params.rc.cascades[cascade];
    return rcFindProbe(params.rc.keys, params.rc.slots, c.tableOffset, c.tableMask, key);
}

// ── Ray sources ──────────────────────────────────────────────────────────

// Whether the source at this pixel traces a ray this frame. Cells near the camera
// cover far more pixels than their probe needs rays, so pixels trace with the
// probability that gives a probe `raysPerProbe` rays: the fixed budget per probe
// that §8.1 suggests. Must be deterministic: the rays are counted before they're
// traced.
public bool rcPixelTracesRay(uint2 pixel, float3 position, uint lod) {
    float distance = max(length(position - params.rc.cameraPosition), params.uniforms.near);
    float cellPixels = params.rc.pixelsPerRadian * rcSpacing(lod, 0) / distance;
    float probability = params.rc.raysPerProbe / (cellPixels * cellPixels);
    return float(xxhash32(uint3(pixel, params.rc.frameIndex))) * (1.0 / 4294967296.0) < probability;
}

public uint rcEncodeNormal(float3 normal) {
    float2 encoded = rcOctEncode(normal);
    uint2 quantized = uint2(saturate(encoded) * 65535.0 + 0.5);
    return quantized.x | (quantized.y << 16);
}

public float3 rcDecodeNormal(uint encoded) {
    return rcOctDecode(float2(encoded & 0xFFFFu, encoded >> 16) / 65535.0);
}

// ── Directions ───────────────────────────────────────────────────────────

// The equal-area spherical mapping (Algorithm 2).
public float3 rcDecodeDirection(float2 uv) {
    float phi = uv.x * 6.2831853071795864;
    float z = 2.0 * uv.y - 1.0;
    float r = sqrt(max(1.0 - z * z, 0.0));
    return float3(r * cos(phi), r * sin(phi), z);
}

// The index of the direction containing `uv` in the top cascade. The directions
// are in Morton order, so a direction's index in cascade n is this shifted right
// by 2 * (RC_CASCADES - 1 - n), and the four directions of cascade n + 1 merged
// into a direction of cascade n are contiguous.
public uint rcDirectionIndex(float2 uv) {
    uint u = min(uint(uv.x * 64.0), 63u);
    uint v = min(uint(uv.y * 32.0), 31u);
    uint index = ((v >> 3) * 8u + (u >> 3)) << 6;
    for (uint bit = 0; bit < 3; bit++) {
        index |= ((u >> bit) & 1u) << (2u * bit);
        index |= ((v >> bit) & 1u) << (2u * bit + 1u);
    }
    return index;
}

// The `index`th point of the R2 low-discrepancy sequence (Roberts 2018), as
// fixed point so that it doesn't lose precision far along the sequence.
public float2 rcR2(uint index) {
    return float2(index * 0xC13FA9A9u, index * 0x91E10DA5u) * (1.0 / 4294967296.0);
}

// ── Irradiance ───────────────────────────────────────────────────────────

// Octahedral mapping of the unit sphere to [0, 1]^2.
public float2 rcOctEncode(float3 n) {
    n /= abs(n.x) + abs(n.y) + abs(n.z);
    float2 folded = n.xy;
    if (n.z < 0.0) {
        folded = (1.0 - abs(n.yx)) * select(n.xy >= 0.0, 1.0, -1.0);
    }
    return folded * 0.5 + 0.5;
}

public float3 rcOctDecode(float2 uv) {
    float2 f = uv * 2.0 - 1.0;
    float3 n = float3(f, 1.0 - abs(f.x) - abs(f.y));
    if (n.z < 0.0) {
        n.xy = (1.0 - abs(n.yx)) * select(n.xy >= 0.0, 1.0, -1.0);
    }
    return normalize(n);
}

// The tile texels of the probes around `position`, weighted trilinearly: mean
// incoming radiance around `normal` in .rgb, multiplied by the fraction of the
// hemisphere known in .a. Probes that don't exist are left out; dividing by .a
// renormalizes for that as well (§4).
public float4 rcSampleIrradiance(
    RcIrradianceSource source,
    Texture2D<float4> atlas,
    uint lod,
    float3 position,
    float3 normal
) {
    float3 grid = position / rcSpacing(lod, 0) - 0.5;
    int3 base = int3(floor(grid));
    float3 f = grid - float3(base);
    float2 oct = rcOctEncode(normal);
    float texel = 1.0 / float(source.tilesPerRow * RC_TILE_SIZE);
    float4 sum = float4(0.0, 0.0, 0.0, 0.0);
    for (uint k = 0; k < 8; k++) {
        uint3 corner = uint3(k & 1u, (k >> 1) & 1u, (k >> 2) & 1u);
        float3 axes = lerp(1.0 - f, f, float3(corner));
        uint key = rcKey(lod, base + int3(corner));
        uint probe = rcFindProbe(source.keys, source.slots, 0, source.tableMask, key);
        if (probe == RC_INVALID) {
            continue;
        }
        float2 tile = float2(probe % source.tilesPerRow, probe / source.tilesPerRow);
        float2 uv = (tile * float(RC_TILE_SIZE) + 1.0 + oct * float(RC_TILE_INNER)) * texel;
        sum += axes.x * axes.y * axes.z * atlas.SampleLevel(params.sky.linearSampler, uv, 0);
    }
    return sum;
}

// `rcSampleIrradiance` with its result made into plain radiance; 0 if nothing is
// known about that hemisphere.
public float3 rcUnpremultiply(float4 irradiance) {
    return irradiance.a > 0.0 ? irradiance.rgb / irradiance.a : float3(0.0, 0.0, 0.0);
}
