// SHARC resolve compute pass.
//
// One thread per cache entry. Each call to SharcResolveEntry() merges the
// per-frame accumulation into the resolved buffer, manages stale-frame
// eviction, and clears the accumulation slot for the next frame
// (AGENTS.md §5.2 / SDK SharcCommon.h docs).
//
// Has its own playout (set 0 binds: 3 SHARC buffers + a small constants UBO);
// the layout is auto-derived by slang_playout from this single source.

#define HASH_GRID_ENABLE_64_BIT_ATOMICS 0
#define HASH_GRID_COMPACT 0
#define SHARC_ENABLE_GLSL 0
#define SHARC_ENABLE_RESPONSIVE_LIGHTING 0
#define SHARC_ENABLE_SH_ENCODING 0
#define SHARC_UPDATE 0
// Keys are discrete voxel identities, not a spatial grid, so SharcResolveEntry's
// adjacent-level reprojection (which reinterprets key bits as grid position /
// level on camera motion) is meaningless and must be off — cellKey does its own
// discrete distance coarsening. MUST match the define in sharc.slang.
#define SHARC_BLEND_ADJACENT_LEVELS 0
#define RW_STRUCTURED_BUFFER(name, type) RWStructuredBuffer<type> name
#define BUFFER_AT_OFFSET(name, offset) name[offset]
#include "SharcCommon.h"

// Layout MUST match `SharcConstantsRaw` in crates/pbr/src/sharc.rs and
// `SharcShaderConstants` in sharc_pt.slang. The resolve shader only uses a
// subset of the fields but the wire format is one shared UBO, so missing /
// reordering fields here causes Resolve to read `entriesNum` from the wrong
// offset and degenerate into a 64-entry no-op.
struct SharcResolveConstants {
    float3 cameraPosition;
    float sceneScale;
    float3 cameraPositionPrev;
    float radianceScale;
    float roughnessMin;
    uint accumulationFrameNum;
    uint staleFrameNum;
    uint entriesNum;
    uint frameIndex;
    uint downscaleFactor;
    uint poolCapacity;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(
    uint3 dispatchThreadId: SV_DispatchThreadID,
    uniform ConstantBuffer<SharcResolveConstants> cb,
    uniform RWStructuredBuffer<HashGridKey> hashEntries,
    uniform RWStructuredBuffer<SharcAccumulationData> accumulation,
    uniform RWStructuredBuffer<SharcPackedData> resolved,
    // Per-slot lock/occupancy buffer (binding 4). SharcResolveEntry resets the
    // flag when it evicts a stale entry so the slot can be reclaimed by the
    // 32-bit-CAS insert path; a harmless no-op for the spinlock path.
    uniform RWStructuredBuffer<uint> lockBuffer,
) {
    uint entryIndex = dispatchThreadId.x;
    if (entryIndex >= cb.entriesNum) return;

    SharcParameters sharc;
    sharc.hashGridParameters.cameraPosition = cb.cameraPosition;
    sharc.hashGridParameters.sceneScale = cb.sceneScale;
    sharc.hashGridParameters.logarithmBase = SHARC_GRID_LOGARITHM_BASE;
    sharc.hashGridParameters.levelBias = SHARC_GRID_LEVEL_BIAS;
    sharc.hashGridData.capacity = cb.entriesNum;
    sharc.hashGridData.hashEntriesBuffer = hashEntries;
#if !HASH_GRID_ENABLE_64_BIT_ATOMICS && !HASH_GRID_COMPACT
    sharc.hashGridData.lockBuffer = lockBuffer;
#endif
    sharc.accumulationBuffer = accumulation;
    sharc.resolvedBuffer = resolved;
    sharc.radianceScale = cb.radianceScale;

    SharcResolveParameters rp;
    rp.cameraPositionPrev = cb.cameraPositionPrev;
    rp.accumulationFrameNum = cb.accumulationFrameNum;
    rp.responsiveFrameNum = cb.accumulationFrameNum;  // unused (no responsive lighting)
    rp.staleFrameNumMax = cb.staleFrameNum;
    rp.frameIndex = cb.frameIndex;

    SharcResolveEntry(entryIndex, sharc, rp);
}
