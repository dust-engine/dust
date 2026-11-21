// SHARC resolve compute pass, one thread per cache slot.
//
// Folds the samples a cell took this frame into its resolved radiance, a running
// average over the last `accumulationFrameNum` frames, moves in the emitter
// reservoir its first Update left, and clears them for the next frame. A cell that goes `staleFrameNum` frames without a sample is
// evicted, which frees its slot for another key.
//
// With SHARC_FLAG_STATS it also sums `SharcStats` over the cells that took
// samples, which `sharc_resolve_pass` reads back and logs.
//
// Has its own playout (set 0 binds the constants, the four cache buffers and
// the stats), auto-derived by slang_playout from this source. `sharc_resolve_pass` in
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
    uniform RWStructuredBuffer<SharcResolvedHot> resolvedHot,
    uniform RWStructuredBuffer<SharcResolvedCold> resolvedCold,
    uniform RWStructuredBuffer<SharcStats> stats,
) {
    uint slot = dispatchThreadId.x;
    HashMap map = HashMap(hashEntries, cb.entriesNum);
    if (slot >= map.capacity || !map.isOccupied(slot)) return;

    SharcAccumulation current = accumulation[slot];
    SharcResolvedCold prevCold = resolvedCold[slot];
    float sampleNum = float(current.sampleNum);

    uint staleFrameNum = sampleNum > 0.0 ? 0 : prevCold.staleFrameNum + 1;
    uint staleFrameNumMax =
        clamp(cb.staleFrameNum, kSharcStaleFrameNumMin, kSharcStaleFrameNumMax);
    if (staleFrameNum >= staleFrameNumMax) {
        map.remove(slot);
        SharcAccumulation empty = {};
        accumulation[slot] = empty;
        resolvedHot[slot] = SharcResolvedHot(float3(0.0, 0.0, 0.0), 0.0);
        SharcReservoir noReservoir = {};
        resolvedCold[slot] = SharcResolvedCold(0, noReservoir);
        return;
    }

    uint accumulatedFrameNum = prevCold.accumulatedFrameNum + 1;
    if (sampleNum == 0.0) {
        // No samples to fold in (and none to clear): only age the cell. Its
        // radiance and reservoir stay as they are, so only the frame counts are
        // written.
        resolvedCold[slot].frameNums =
            SharcResolvedCold::packFrameNums(accumulatedFrameNum, staleFrameNum);
        return;
    }

    // Past the accumulation window, weight the history as that many frames'
    // worth of samples, so older frames fade out of the average.
    SharcResolvedHot prevHot = resolvedHot[slot];
    float prevSampleNum = prevHot.sampleNum;
    uint accumulationFrameNum = clamp(cb.accumulationFrameNum, kSharcAccumulationFrameNumMin,
                                      kSharcAccumulationFrameNumMax);
    if (accumulatedFrameNum > accumulationFrameNum) {
        prevSampleNum *= float(accumulationFrameNum) / float(accumulatedFrameNum);
        accumulatedFrameNum = accumulationFrameNum;
    }

    float totalSampleNum = prevSampleNum + sampleNum;
    float3 radiance = (prevHot.radiance * prevSampleNum + current.radiance) / totalSampleNum;
    SharcResolvedHot hot = SharcResolvedHot(radiance, totalSampleNum);
    resolvedHot[slot] = hot;
    // The reservoir the cell's first Update left (none when the Update doesn't
    // sample emitters).
    resolvedCold[slot] = SharcResolvedCold(
        SharcResolvedCold::packFrameNums(accumulatedFrameNum, 0), current.reservoir);

    if ((cb.flags & SHARC_FLAG_STATS) != 0) {
        // As stored, in half precision. A cell new this frame has no change.
        float luminance = dot(hot.radiance, kRec709Luminance);
        float prevLuminance =
            prevHot.sampleNum > 0.0 ? dot(prevHot.radiance, kRec709Luminance) : luminance;
        InterlockedAdd(stats[0].luminanceSum, luminance);
        InterlockedAdd(stats[0].changeSum, abs(luminance - prevLuminance));
        InterlockedAdd(stats[0].cellCount, 1u);
    }

    SharcAccumulation empty = {};
    accumulation[slot] = empty;
}
