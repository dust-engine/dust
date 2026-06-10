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
    if (debugMode == SharcDebugMode::None) {
        return;
    }
    float brightness = params.sharc.g_sharcConstants.debugBrightness;

    // Occupancy is a screen-space view of the table itself — no G-buffer needed.
    if (debugMode == SharcDebugMode::HashOccupancy) {
        float3 occ = params.sharc.debugOccupancy(px, dims);
        // Empty cells / inter-cell borders come back black: leave those pixels
        // as the rendered image so only populated table cells draw on top.
        if (any(occ != float3(0.0, 0.0, 0.0))) {
            params.gbuffer.occlusionTexture[px] = float4(occ * brightness, 1.0);
        }
        return;
    }

    // Per-surface views. The G-buffers are render-resolution while
    // occlusionTexture is display-resolution (they differ under DLSS
    // upscaling), so map this display pixel onto its render-res G-buffer texel
    // with nearest sampling.
    uint2 renderDims;
    params.gbuffer.depthTexture.GetDimensions(renderDims.x, renderDims.y);
    uint2 gpx = min(
        uint2((float2(px) + 0.5) * float2(renderDims) / float2(dims)),
        renderDims - uint2(1, 1));

    // depth == 0 ⇒ sky / UI (no surface): leave the rendered pixel untouched, so
    // the overlay only covers geometry — matching the old closest-hit behavior.
    if (params.gbuffer.depthTexture[gpx] == 0.0) {
        return;
    }
    float3 worldPos = params.pixelPositionWorldSpace(gpx, renderDims);
    float3 worldNormal = params.gbuffer.normalTexture[gpx].xyz;

    float3 debugColor = float3(0.0, 0.0, 0.0);
    switch (debugMode) {
    case SharcDebugMode::ColoredHash:
        // Colored hash of the grid cell this surface point lands in.
        debugColor = params.sharc.debugColoredHash(worldPos, worldNormal);
        break;
    case SharcDebugMode::CachedRadiance:
        // Resolved radiance currently cached here; misses paint black. Written
        // straight to the SDR target (no tonemap), so use the brightness slider
        // to bring HDR values into range.
        float3 cached;
        debugColor = params.sharc.getCachedRadiance(worldPos, worldNormal, cached)
                         ? cached
                         : float3(0.0, 0.0, 0.0);
        break;
    case SharcDebugMode::HashCollisions:
        // Hash-collision heatmap: linear-probe bucket offset at this point
        // (blue = no collision … red = heavy probing).
        debugColor = params.sharc.debugHashCollisions(worldPos, worldNormal);
        break;
    }
    params.gbuffer.occlusionTexture[px] = float4(debugColor * brightness, 1.0);
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
