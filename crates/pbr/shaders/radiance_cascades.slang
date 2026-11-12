// Radiance cascades: diffuse global illumination from radiance probes placed in
// world space. So far this builds the probes and paints them in the debug
// views; nothing lights the frame from them yet, and a probe is only the key
// of its cell.
//
// The probes come in cascades: going up a cascade doubles the spacing between
// them. Probes exist only where visible surfaces need them. Each cascade
// divides space into a grid of cells, one probe per cell, and keeps a hash
// table of the cells that need one. The cells double in size each time the
// distance from the camera doubles (the levels of detail, LODs), which keeps
// their size on screen, and the probe count, about constant.
//
// Each frame:
//   1. Insert the cascade 0 cell of every visible surface point.
//   2. Give each of a cascade's cells a probe, and insert each probe's parent:
//      the cell of the next cascade, at the same LOD, that contains it. Repeat
//      up the cascades.
//
// A key packs a cell's LOD and coordinates into 32 bits, since 64-bit atomics
// can't be assumed: 5 bits for the LOD plus one, so that no key is 0, and 9 per
// axis, wrapped. Wrapping loses nothing because the cells of a LOD span fewer
// than 512 per axis around the camera; RadianceCascadesConfig keeps it so.

import pbr_common;

// No probe, or no slot.
static const uint RC_INVALID = 0xFFFFFFFFu;
// Linear probing gives up after this many slots, dropping the insertion.
static const uint RC_MAX_PROBE_STEPS = 32;
// The highest LOD a key can hold. Everything farther shares it.
static const uint RC_MAX_LOD = 30;

// The push constants of the build's passes, after the ray generation stage's
// range (pbr.playout.ron).
struct RcPush {
    // The cascade a pass works on.
    [vk::offset(4)] uint cascade;
}

[vk::push_constant] uniform RcPush pc;

// ── Levels of detail ─────────────────────────────────────────────────────

// The LOD of a point: 0 within lodDistance of the camera, and one more each
// time the distance doubles. The distance is Chebyshev, so that a LOD's cells
// around the camera span a cube.
uint rcLodOf(float3 position) {
    float3 offset = abs(position - params.uniforms.cameraOrigin());
    float distance = max(offset.x, max(offset.y, offset.z)) / params.rc.lodDistance;
    return distance < 1.0 ? 0u : min(uint(floor(log2(distance))) + 1u, RC_MAX_LOD);
}

// World units between the probes of a cascade at a LOD: each LOD and each
// cascade up doubles them.
float rcSpacing(uint lod, uint cascade) {
    return params.rc.spacing * exp2(float(lod + cascade));
}

// Where a surface point counts as being on the grid: a twentieth of a cell off
// the surface, so that voxel faces, which often lie exactly on a cell boundary,
// consistently land in the cell they face.
float3 rcGridPoint(float3 position, float3 normal, uint lod) {
    return position + normal * 0.05 * rcSpacing(lod, 0);
}

// ── Hash tables ──────────────────────────────────────────────────────────

// The cell of a cascade's grid at a LOD that holds `position`.
int3 rcCellOf(float3 position, uint lod, uint cascade) {
    return int3(floor(position / rcSpacing(lod, cascade)));
}

uint rcKey(uint lod, int3 cell) {
    uint3 wrapped = uint3(cell) & 511u;
    return ((lod + 1u) << 27) | (wrapped.x << 18) | (wrapped.y << 9) | wrapped.z;
}

// The cell behind a key: the one with the wrapped coordinates nearest the camera.
int3 rcCellOfKey(uint key, uint cascade, out uint lod) {
    lod = (key >> 27) - 1u;
    int3 wrapped = int3((key >> 18) & 511u, (key >> 9) & 511u, key & 511u);
    int3 center = rcCellOf(params.uniforms.cameraOrigin(), lod, cascade);
    int3 difference = (wrapped - center) & 511;
    return center + (difference ^ 256) - 256;
}

// Mixes a key's bits (MurmurHash3's finalizer), so that neighboring cells land
// in scattered slots.
uint rcHash(uint key) {
    key ^= key >> 16;
    key *= 0x85EBCA6Bu;
    key ^= key >> 13;
    key *= 0xC2B2AE35u;
    return key ^ (key >> 16);
}

// Adds a key to a cascade's table, unless it's there already.
void rcInsert(uint cascade, uint key) {
    RcCascade c = params.rc.cascades[cascade];
    uint hash = rcHash(key);
    for (uint i = 0; i < RC_MAX_PROBE_STEPS; i++) {
        uint slot = c.tableOffset + ((hash + i) & c.tableMask);
        uint current = params.rc.keys[slot];
        if (current == 0) {
            InterlockedCompareExchange(params.rc.keys[slot], 0u, key, current);
            if (current == 0) {
                return;
            }
        }
        if (current == key) {
            return;
        }
    }
}

// The index of the probe of a key in a cascade, once its probes are allocated;
// RC_INVALID if it has none.
uint rcFindProbe(uint cascade, uint key) {
    RcCascade c = params.rc.cascades[cascade];
    uint hash = rcHash(key);
    for (uint i = 0; i < RC_MAX_PROBE_STEPS; i++) {
        uint slot = c.tableOffset + ((hash + i) & c.tableMask);
        uint current = params.rc.keys[slot];
        if (current == key) {
            return params.rc.slots[slot];
        }
        if (current == 0) {
            break;
        }
    }
    return RC_INVALID;
}

// ── Building the probes: steps 1 and 2 ───────────────────────────────────

// Step 1, over the G-buffer's pixels.
[shader("compute")]
[numthreads(8, 8, 1)]
void rcInsertPixels(uint3 threadId: SV_DispatchThreadID) {
    uint2 pixel = threadId.xy;
    if (any(pixel >= params.rc.renderExtent) || !params.gbuffer.hasSurface(pixel)) {
        return;
    }
    float3 position = params.pixelPositionWorldSpace(pixel, params.rc.renderExtent);
    float3 normal = params.gbuffer.normalTexture[pixel].xyz;
    uint lod = rcLodOf(position);
    float3 point = rcGridPoint(position, normal, lod);
    rcInsert(0, rcKey(lod, rcCellOf(point, lod, 0)));
}

// Gives the cells in `pc.cascade`'s table probes, while the cascade has room.
// Over the table's slots.
[shader("compute")]
[numthreads(64, 1, 1)]
void rcAllocate(uint3 threadId: SV_DispatchThreadID) {
    RcCascade c = params.rc.cascades[pc.cascade];
    if (threadId.x > c.tableMask) {
        return;
    }
    uint slot = c.tableOffset + threadId.x;
    uint key = params.rc.keys[slot];
    if (key == 0) {
        return;
    }
    uint index;
    InterlockedAdd(params.rc.counters[RC_COUNTER_PROBES + pc.cascade], 1u, index);
    if (index >= c.probeCapacity) {
        params.rc.slots[slot] = RC_INVALID;
        return;
    }
    params.rc.slots[slot] = index;
    params.rc.probes[c.probeOffset + index] = key;
}

// Clamps `pc.cascade`'s probe count to its capacity, since rcAllocate counts
// the cells it had no room for too, and writes the arguments of the indirect
// dispatches over its probes. One thread.
[shader("compute")]
[numthreads(1, 1, 1)]
void rcWriteArgs() {
    uint cascade = pc.cascade;
    uint count = min(params.rc.counters[RC_COUNTER_PROBES + cascade], params.rc.cascades[cascade].probeCapacity);
    uint probeArgs = RC_COUNTER_PROBE_ARGS + cascade * 3u;
    params.rc.counters[RC_COUNTER_PROBES + cascade] = count;
    params.rc.counters[probeArgs] = (count + 63u) / 64u;
    params.rc.counters[probeArgs + 1u] = 1u;
    params.rc.counters[probeArgs + 2u] = 1u;
}

// Inserts the parent of each probe of the cascade below `pc.cascade` into
// `pc.cascade`'s table. Over those probes.
[shader("compute")]
[numthreads(64, 1, 1)]
void rcInsertParents(uint3 threadId: SV_DispatchThreadID) {
    uint cascade = pc.cascade;
    if (threadId.x >= params.rc.counters[RC_COUNTER_PROBES + cascade - 1u]) {
        return;
    }
    uint childKey = params.rc.probes[params.rc.cascades[cascade - 1u].probeOffset + threadId.x];
    uint lod;
    int3 cell = rcCellOfKey(childKey, cascade - 1u, lod);
    // The next cascade's cells are twice as wide, so halving the coordinates,
    // rounding down, gives the parent.
    rcInsert(cascade, rcKey(lod, cell >> 1));
}

// ── Debug views ──────────────────────────────────────────────────────────

// Values of RcConstants::debugMode (RadianceCascadesDebugMode in
// radiance_cascades.rs). 0 paints nothing.
static const uint RC_DEBUG_PROBES = 1;
static const uint RC_DEBUG_LOD = 2;

// A color for each value of `id`. Multiplying by an odd constant near 2^32 over
// the golden ratio scatters neighboring ids across the low 24 bits, which
// become the color.
float3 colorOf(uint id) {
    uint hash = id * 0x9E3779B1u;
    return float3(hash & 0xFFu, (hash >> 8) & 0xFFu, (hash >> 16) & 0xFFu) / 255.0;
}

// Paints the debug view over the debug overlay, at display resolution; tonemap
// composites it.
[shader("compute")]
[numthreads(8, 8, 1)]
void rcDebugOverlay(uint3 threadId: SV_DispatchThreadID) {
    uint2 px = threadId.xy;
    uint2 displayDims;
    params.gbuffer.debugOverlayTexture.GetDimensions(displayDims.x, displayDims.y);
    if (any(px >= displayDims)) {
        return;
    }
    // The G-buffer pixel under this one.
    uint2 renderExtent = params.rc.renderExtent;
    uint2 pixel = min(uint2((float2(px) + 0.5) / float2(displayDims) * float2(renderExtent)), renderExtent - 1u);
    if (!params.gbuffer.hasSurface(pixel)) {
        return;
    }
    float3 position = params.pixelPositionWorldSpace(pixel, renderExtent);
    uint lod = rcLodOf(position);

    float3 color;
    switch (params.rc.debugMode) {
    case RC_DEBUG_PROBES: {
        float3 normal = params.gbuffer.normalTexture[pixel].xyz;
        uint key = rcKey(lod, rcCellOf(rcGridPoint(position, normal, lod), lod, 0));
        // White where the cell got no probe: cascade 0 was full.
        color = rcFindProbe(0, key) == RC_INVALID ? float3(1.0, 1.0, 1.0) : colorOf(key);
        break;
    }
    case RC_DEBUG_LOD:
        // Plus one, since 0 is black.
        color = colorOf(lod + 1u);
        break;
    default:
        return;
    }
    params.gbuffer.writeOverlay(px, color);
}
