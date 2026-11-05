// Radiance cascades: lights the visible surfaces from the cache (the end of step 6
// in rc_common.slang), and paints its debug views.

import pbr_common;
import rc_common;

// The cache's light at a surface point, blended across a LOD overlap (§4.1).
float4 sampleSurface(float3 position, float3 normal, out RcLod lod) {
    lod = rcLodOf(position);
    uint level = lod.level + params.rc.lodOffset;
    float4 irradiance = rcSampleIrradiance(
        params.rc.own, params.rcIrradianceAtlas, level, rcGridPoint(position, normal, level), normal);
    if (lod.blend > 0.0) {
        uint coarser = level + 1u;
        float4 other = rcSampleIrradiance(
            params.rc.own, params.rcIrradianceAtlas, coarser, rcGridPoint(position, normal, coarser), normal);
        irradiance = lerp(irradiance, other, lod.blend);
    }
    return irradiance;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void rcShade(uint3 threadId: SV_DispatchThreadID) {
    uint2 pixel = threadId.xy;
    if (any(pixel >= params.rc.renderExtent) || !rcIsSurfacePixel(pixel)) {
        return;
    }
    float3 position = params.pixelPositionWorldSpace(pixel, params.rc.renderExtent);
    float3 normal = params.gbuffer.normalTexture[pixel].xyz;
    RcLod lod;
    float3 radiance = rcUnpremultiply(sampleSurface(position, normal, lod));
    // Lambertian: albedo / π times irradiance, π times mean radiance.
    float3 albedo = params.gbuffer.albedoSRGB[pixel].rgb;
    float4 current = params.outputTexture[pixel];
    params.outputTexture[pixel] = float4(current.rgb + albedo * radiance, current.a);
}

// Values of `RcConstants::debugMode` (RadianceCascadesDebugMode in
// radiance_cascades.rs).
static const uint RC_DEBUG_IRRADIANCE = 1;
static const uint RC_DEBUG_PROBES = 2;
static const uint RC_DEBUG_LOD = 3;
static const uint RC_DEBUG_COVERAGE = 4;
static const uint RC_DEBUG_BOUNCE = 5;

float3 colorOfHash(uint hash) {
    return float3(hash & 0xFFu, (hash >> 8) & 0xFFu, (hash >> 16) & 0xFFu) / 255.0;
}

// Dispatched at display resolution over the overlay layer, which tonemap
// composites.
[shader("compute")]
[numthreads(8, 8, 1)]
void rcDebugOverlay(uint3 threadId: SV_DispatchThreadID) {
    uint2 px = threadId.xy;
    uint2 displayDims;
    params.gbuffer.debugOverlayTexture.GetDimensions(displayDims.x, displayDims.y);
    if (any(px >= displayDims)) {
        return;
    }
    uint2 screenSize = params.rc.renderExtent;
    uint2 pixel = min(uint2((float2(px) + 0.5) / float2(displayDims) * float2(screenSize)), screenSize - 1u);
    if (!rcIsSurfacePixel(pixel)) {
        return;
    }
    float3 position = params.pixelPositionWorldSpace(pixel, screenSize);
    float3 normal = params.gbuffer.normalTexture[pixel].xyz;
    RcLod lod;
    float4 irradiance = sampleSurface(position, normal, lod);
    uint level = lod.level + params.rc.lodOffset;

    float3 color = float3(0.0, 0.0, 0.0);
    switch (params.rc.debugMode) {
    case RC_DEBUG_IRRADIANCE:
        color = rcUnpremultiply(irradiance) * params.rc.debugScale;
        break;
    case RC_DEBUG_PROBES: {
        float3 point = rcGridPoint(position, normal, level);
        uint key = rcKey(level, rcCellOf(point, level, 0));
        color = colorOfHash(key * 0x9E3779B1u);
        // Cells without a probe stand out in white.
        if (rcFindProbe(0, key) == RC_INVALID) {
            color = float3(1.0, 1.0, 1.0);
        }
        break;
    }
    case RC_DEBUG_LOD:
        color = colorOfHash((lod.level + 1u) * 0x9E3779B1u) * (1.0 - 0.5 * lod.blend);
        break;
    case RC_DEBUG_COVERAGE:
        color = irradiance.aaa;
        break;
    case RC_DEBUG_BOUNCE:
        // Black without a secondary cache.
        if ((params.rc.flags & RC_FLAG_BOUNCE) != 0) {
            uint bounceLevel = lod.level + params.rc.bounce.lodOffset;
            float4 bounced = rcSampleIrradiance(
                params.rc.bounce, params.rcBounceAtlas, bounceLevel, rcGridPoint(position, normal, bounceLevel), normal);
            color = rcUnpremultiply(bounced) * params.rc.debugScale;
        }
        break;
    default:
        return;
    }
    params.gbuffer.writeOverlay(px, color);
}
