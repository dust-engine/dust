// Radiance cascades: this frame's probes and their share of the rays (steps 1 to
// 3 in rc_common.slang), as compute passes over the source points and the probes.

import pbr_common;
import rc_common;
import random;

[vk::push_constant] uniform RcPush pc;

// The low 24 bits of a slot count the rays of its cell until `rcAllocate`; the
// high 8 hold the age of a probe kept from last frame.
static const uint RC_SLOT_RAY_MASK = 0xFFFFFFu;
static const uint RC_SLOT_AGE_SHIFT = 24;

// Every c0 cell with a surface point gets a probe; only rays need counting.
void insertSource(float3 position, float3 normal, uint lod, bool tracesRay) {
    float3 point = rcGridPoint(position, normal, lod);
    uint slot = rcInsert(0, rcKey(lod, rcCellOf(point, lod, 0)));
    if (tracesRay && slot != RC_INVALID) {
        InterlockedAdd(params.rc.slots[slot], 1u);
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void rcInsertPixels(uint3 threadId: SV_DispatchThreadID) {
    uint2 pixel = threadId.xy;
    if (any(pixel >= params.rc.renderExtent) || !rcIsSurfacePixel(pixel)) {
        return;
    }
    float3 position = params.pixelPositionWorldSpace(pixel, params.rc.renderExtent);
    float3 normal = params.gbuffer.normalTexture[pixel].xyz;
    RcLod lod = rcLodOf(position);
    uint level = lod.level + params.rc.lodOffset;
    insertSource(position, normal, level, rcPixelTracesRay(pixel, position, level));
    // Rays follow the sequence of the finer LOD, and feed the coarser one too.
    if (lod.blend > 0.0) {
        insertSource(position, normal, level + 1u, false);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void rcInsertHits(uint3 threadId: SV_DispatchThreadID) {
    if (threadId.x >= params.rc.hitCounters[RC_HIT_COUNT]) {
        return;
    }
    RcHit hit = params.rc.hits[threadId.x];
    uint level = rcLodOf(hit.position).level + params.rc.lodOffset;
    insertSource(hit.position, rcDecodeNormal(hit.normal), level, true);
}

// Last frame's probes get another frame, so that one that no ray left from this
// frame keeps its history. Matters to the secondary cache, whose source points
// differ every frame. Runs before the sources, over last frame's cascade 0.
[shader("compute")]
[numthreads(64, 1, 1)]
void rcCarryOver(uint3 threadId: SV_DispatchThreadID) {
    if (threadId.x >= params.rc.prevCounters[RC_COUNTER_PROBES]) {
        return;
    }
    RcProbe probe = params.rc.prevProbes[threadId.x];
    if (probe.age >= params.rc.maxAge) {
        return;
    }
    uint slot = rcInsert(0, probe.key);
    if (slot != RC_INVALID) {
        InterlockedOr(params.rc.slots[slot], (probe.age + 1u) << RC_SLOT_AGE_SHIFT);
    }
}

// Gives the table's cells probes (dispatched over the table's slots).
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
    uint packed = params.rc.slots[slot];
    uint index;
    InterlockedAdd(params.rc.counters[RC_COUNTER_PROBES + pc.cascade], 1u, index);
    if (index >= c.probeCapacity) {
        params.rc.slots[slot] = RC_INVALID;
        return;
    }
    params.rc.slots[slot] = index;
    RcProbe probe;
    probe.key = key;
    // A temporal alpha of 1 means single-frame lighting, with no history at all.
    probe.prevIndex = params.rc.temporalAlpha < 1.0
        ? rcFindProbe(params.rc.prevKeys, params.rc.prevSlots, c.tableOffset, c.tableMask, key)
        : RC_INVALID;
    probe.parent = RC_INVALID;
    probe.rayCount = packed & RC_SLOT_RAY_MASK;
    probe.rayCursor = 0;
    probe.age = probe.rayCount > 0 ? 0u : packed >> RC_SLOT_AGE_SHIFT;
    params.rc.probes[c.probeOffset + index] = probe;
}

// Clamps a cascade's probe count to its capacity, and turns it into the indirect
// dispatch arguments of the passes over its probes.
[shader("compute")]
[numthreads(1, 1, 1)]
void rcWriteArgs() {
    uint cascade = pc.cascade;
    uint count = min(params.rc.counters[RC_COUNTER_PROBES + cascade], params.rc.cascades[cascade].probeCapacity);
    // `rcMerge` outputs one value per direction of the cascade below.
    uint outputs = rcDirectionCount(cascade == 0 ? 0u : cascade - 1u);
    uint probeArgs = RC_COUNTER_PROBE_ARGS + cascade * 3u;
    uint mergeArgs = RC_COUNTER_MERGE_ARGS + cascade * 3u;
    params.rc.counters[RC_COUNTER_PROBES + cascade] = count;
    params.rc.counters[probeArgs] = (count + 63u) / 64u;
    params.rc.counters[probeArgs + 1u] = 1u;
    params.rc.counters[probeArgs + 2u] = 1u;
    params.rc.counters[mergeArgs] = count * (outputs / 32u);
    params.rc.counters[mergeArgs + 1u] = 1u;
    params.rc.counters[mergeArgs + 2u] = 1u;
}

// Dispatched over the probes of the cascade below `pc.cascade`: each one's parent
// cell needs a probe, with all the rays of its children (Algorithm 3, lines 7-9).
[shader("compute")]
[numthreads(64, 1, 1)]
void rcInsertParents(uint3 threadId: SV_DispatchThreadID) {
    uint cascade = pc.cascade;
    if (threadId.x >= params.rc.counters[RC_COUNTER_PROBES + cascade - 1u]) {
        return;
    }
    uint childIndex = params.rc.cascades[cascade - 1u].probeOffset + threadId.x;
    RcProbe child = params.rc.probes[childIndex];
    uint lod;
    int3 cell = rcCellOfKey(child.key, cascade - 1u, lod);
    uint slot = rcInsert(cascade, rcKey(lod, cell >> 1));
    if (slot != RC_INVALID && child.rayCount > 0) {
        InterlockedAdd(params.rc.slots[params.rc.cascades[cascade].tableOffset + slot], child.rayCount);
    }
    params.rc.probes[childIndex].parent = slot;
}

// Run from the top cascade down, since a probe takes its range of the R2 sequence
// out of its parent's (Algorithm 3, lines 12-20). Also finds the probes of the
// next cascade around this one that `rcMerge` interpolates. A probe's coordinates
// are half-odd multiples of half its parent's spacing, so these are its parent
// and the parent's neighbors on the probe's side; the weights are fixed.
[shader("compute")]
[numthreads(64, 1, 1)]
void rcLink(uint3 threadId: SV_DispatchThreadID) {
    uint cascade = pc.cascade;
    if (threadId.x >= params.rc.counters[RC_COUNTER_PROBES + cascade]) {
        return;
    }
    RcCascade c = params.rc.cascades[cascade];
    uint index = c.probeOffset + threadId.x;
    RcProbe probe = params.rc.probes[index];
    uint offset = 0;
    if (cascade == RC_CASCADES - 1u) {
        InterlockedAdd(params.rc.counters[RC_COUNTER_RAYS], probe.rayCount, offset);
    } else {
        RcCascade up = params.rc.cascades[cascade + 1u];
        uint parent = RC_INVALID;
        if (probe.parent != RC_INVALID) {
            parent = params.rc.slots[up.tableOffset + probe.parent];
        }
        params.rc.probes[index].parent = parent;
        if (parent != RC_INVALID) {
            InterlockedAdd(params.rc.probes[up.probeOffset + parent].rayCursor, probe.rayCount, offset);
        }
        uint lod;
        int3 cell = rcCellOfKey(probe.key, cascade, lod);
        int3 side = (cell & 1) * 2 - 1;
        params.rc.neighbors[c.neighborOffset + threadId.x * 8u] = parent;
        for (uint k = 1; k < 8; k++) {
            int3 corner = (cell >> 1) + int3(k & 1u, (k >> 1) & 1u, (k >> 2) & 1u) * side;
            params.rc.neighbors[c.neighborOffset + threadId.x * 8u + k] = rcFindProbe(cascade + 1u, rcKey(lod, corner));
        }
    }
    params.rc.probes[index].rayCursor = offset;
}

// Clamps the number of ray hits kept this frame to the capacity of their list, and
// turns it into the indirect arguments of next frame's secondary cache.
[shader("compute")]
[numthreads(1, 1, 1)]
void rcWriteHitArgs() {
    uint count = min(params.rc.hitCounters[RC_HIT_COUNT], params.rc.hitCapacity);
    params.rc.hitCounters[RC_HIT_COUNT] = count;
    params.rc.hitCounters[RC_HIT_BUILD_ARGS] = (count + 63u) / 64u;
    params.rc.hitCounters[RC_HIT_BUILD_ARGS + 1u] = 1u;
    params.rc.hitCounters[RC_HIT_BUILD_ARGS + 2u] = 1u;
    params.rc.hitCounters[RC_HIT_TRACE_ARGS] = count;
    params.rc.hitCounters[RC_HIT_TRACE_ARGS + 1u] = 1u;
    params.rc.hitCounters[RC_HIT_TRACE_ARGS + 2u] = 1u;
}
