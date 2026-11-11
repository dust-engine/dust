// Radiance cascades: paints the debug views over the frame.

import pbr_common;
import rc_common;

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

// Over the debug overlay, at display resolution; tonemap composites it.
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
