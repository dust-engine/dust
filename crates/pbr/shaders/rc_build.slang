// Radiance cascades: builds this frame's probes (steps 1 and 2 in
// rc_common.slang).

import pbr_common;
import rc_common;

[vk::push_constant] uniform RcPush pc;

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
    RcProbe probe;
    probe.key = key;
    params.rc.probes[c.probeOffset + index] = probe;
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
    RcProbe child = params.rc.probes[params.rc.cascades[cascade - 1u].probeOffset + threadId.x];
    uint lod;
    int3 cell = rcCellOfKey(child.key, cascade - 1u, lod);
    // The next cascade's cells are twice as wide, so halving the coordinates,
    // rounding down, gives the parent.
    rcInsert(cascade, rcKey(lod, cell >> 1));
}
