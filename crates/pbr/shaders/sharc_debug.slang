import pbr_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 threadId: SV_DispatchThreadID) {
    // Dispatched at display resolution: occlusionTexture == the SDR target.
    uint2 px = threadId.xy;
    uint2 dims;
    params.gbuffer.occlusionTexture.GetDimensions(dims.x, dims.y);
    if (any(px >= dims)) {
        return;
    }
    if (params.gbuffer.occlusionTexture[px].a == 1.0) {
        // Skip existing GUI
        return;
    }

    SharcDebugMode debugMode = params.sharc.g_sharcConstants.debugMode;
    // The per-surface views (colored hash / cached radiance / hash collisions)
    // key the cache by voxel identity, which only the primary closest-hit can
    // compute (vox_pbr.slang) — it can't be recovered from the G-buffer here.
    // So this screen-space pass now owns only the table-occupancy view;
    // CandidatePool has its own entry point below.
    if (debugMode != SharcDebugMode::HashOccupancy) {
        return;
    }
    float brightness = params.sharc.g_sharcConstants.debugBrightness;

    // Occupancy is a screen-space view of the table itself — no G-buffer needed.
    float3 occ = params.sharc.debugOccupancy(px, dims);
    // Empty cells / inter-cell borders come back black: leave those pixels
    // as the rendered image so only populated table cells draw on top.
    if (any(occ != float3(0.0, 0.0, 0.0))) {
        params.gbuffer.occlusionTexture[px] = float4(occ * brightness, 1.0);
    }
}

// Candidate-pool splat (SharcDebugMode::CandidatePool). This is a *scatter*, not
// a per-pixel pass: one thread per pool slot, dispatched over the pool rather
// than the screen. Each occupied slot's candidate is projected to the current
// camera and marked, so you can see where the cache is generating candidates —
// i.e. where it's currently short on samples. The host binds the side
// final-gather wrote this frame to the read slots the accessors sample.
[shader("compute")]
[numthreads(32, 1, 1)]
void splatCandidates(uint3 threadId: SV_DispatchThreadID) {
    uint slot = threadId.x;
    if (slot >= params.sharc.g_sharcConstants.poolCapacity) {
        return;
    }
    if (params.sharc.keyAt(slot) == 0u) {
        return; // empty slot
    }
    Candidate c = params.sharc.candidateAt(slot);

    uint2 dims;
    params.gbuffer.occlusionTexture.GetDimensions(dims.x, dims.y);
    float2 pixelF;
    if (!params.uniforms.projectToPixel(c.worldPos, dims, pixelF)) {
        return; // behind the camera
    }
    int2 center = int2(floor(pixelF));
    float3 color = float3(0.1, 1.0, 0.2) * params.sharc.g_sharcConstants.debugBrightness;

    if (any(center >= int2(0, 0)) && any(center < int2(dims))) {
        if (params.gbuffer.occlusionTexture[uint2(center)].a != 1.0) {
            // don't paint over egui
            params.gbuffer.occlusionTexture[uint2(center)] = float4(color, 1.0);
        }
    }
}
