// SHARC resolve compute pass, one thread per cache slot.
//
// Folds the samples a cell took this frame into its resolved radiance, a running
// average over the last `accumulationFrameNum` frames, and clears them for the
// next frame. A cell that goes `staleFrameNum` frames without a sample is
// evicted, which frees its slot for another key.
//
// Has its own playout (set 0 binds the constants and the three cache buffers),
// auto-derived by slang_playout from this source. `sharc_resolve_pass` in
// crates/pbr/src/sharc.rs writes the bindings by number, in parameter order.

import hash_map;
import sharc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(
    uint3 dispatchThreadId: SV_DispatchThreadID,
    uniform ConstantBuffer<SharcShaderConstants> cb,
    uniform RWStructuredBuffer<uint> hashEntries,
    uniform RWStructuredBuffer<SharcAccumulation> accumulation,
    uniform RWStructuredBuffer<SharcResolved> resolved,
) {
    uint slot = dispatchThreadId.x;
    HashMap map = HashMap(hashEntries, cb.entriesNum);
    if (slot >= map.capacity || !map.isOccupied(slot)) return;

    SharcAccumulation current = accumulation[slot];
    SharcResolved prev = resolved[slot];
    float sampleNum = float(current.sampleNum);

    uint staleFrameNum = sampleNum > 0.0 ? 0 : prev.staleFrameNum + 1;
    uint staleFrameNumMax =
        clamp(cb.staleFrameNum, kSharcStaleFrameNumMin, kSharcStaleFrameNumMax);
    if (staleFrameNum >= staleFrameNumMax) {
        map.remove(slot);
        SharcAccumulation empty = {};
        accumulation[slot] = empty;
        resolved[slot] = SharcResolved(float3(0.0, 0.0, 0.0), 0.0, 0, 0);
        return;
    }

    uint accumulatedFrameNum = prev.accumulatedFrameNum + 1;
    if (sampleNum == 0.0) {
        // No samples to fold in (and none to clear): only age the cell.
        resolved[slot] =
            SharcResolved(prev.radiance, prev.sampleNum, accumulatedFrameNum, staleFrameNum);
        return;
    }

    // Past the accumulation window, weight the history as that many frames'
    // worth of samples, so older frames fade out of the average.
    float prevSampleNum = prev.sampleNum;
    uint accumulationFrameNum = clamp(cb.accumulationFrameNum, kSharcAccumulationFrameNumMin,
                                      kSharcAccumulationFrameNumMax);
    if (accumulatedFrameNum > accumulationFrameNum) {
        prevSampleNum *= float(accumulationFrameNum) / float(accumulatedFrameNum);
        accumulatedFrameNum = accumulationFrameNum;
    }

    float totalSampleNum = prevSampleNum + sampleNum;
    float3 radiance = (prev.radiance * prevSampleNum + current.radiance) / totalSampleNum;
    resolved[slot] = SharcResolved(radiance, totalSampleNum, accumulatedFrameNum, 0);

    SharcAccumulation empty = {};
    accumulation[slot] = empty;
}
