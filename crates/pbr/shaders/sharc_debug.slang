// Debug views of the radiance cache (sharc.slang), drawn over the frame: one
// thread per texel of the debug overlay, which has display resolution.

import pbr_common;
import sharc;

// Values of `view`.
static const uint kViewGrid = 1;
static const uint kViewRadiance = 2;
static const uint kViewOccupancy = 3;
static const uint kViewBounces = 4;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(
    uint3 id : SV_DispatchThreadID,
    uniform SharcConstants constants,
    // For the view of them, how many bounces each pixel's path took, row after
    // row (path_tracer.slang).
    uniform uint *bounces,
    uniform uint view
) {
    uint2 texel = id.xy;
    uint2 overlaySize;
    params.gbuffer.debugOverlayTexture.GetDimensions(overlaySize.x, overlaySize.y);
    if (any(texel >= overlaySize)) {
        return;
    }
    SharcParameters cache = constants.parameters();

    if (view == kViewOccupancy) {
        float3 color = HashGridDebugOccupancy(texel, overlaySize, cache.hashGridData);
        if (any(color > 0.0)) {
            params.gbuffer.writeOverlay(texel, color);
        }
        return;
    }

    // The surface of the pixel that the texel is in.
    uint2 renderSize;
    params.gbuffer.depthTexture.GetDimensions(renderSize.x, renderSize.y);
    uint2 pixel = texel * renderSize / overlaySize;
    // The sky.
    if (params.gbuffer.depthTexture[pixel] == 0.0) {
        return;
    }
    if (view == kViewBounces) {
        bool wentOn = bounces[pixel.y * renderSize.x + pixel.x] > 0;
        params.gbuffer.writeOverlay(texel, wentOn ? float3(1.0, 0.0, 0.0) : float3(0.0, 1.0, 0.0));
        return;
    }
    float3 position = params.pixelPositionWorldSpace(pixel, renderSize);
    // The G-buffers don't have its emission.
    SharcHitData surface = sharcSurface(
        cache,
        position,
        params.gbuffer.normalTexture[pixel].xyz,
        params.gbuffer.albedoSRGB[pixel].rgb,
        float3(0.0)
    );

    float3 color = float3(0.0);
    if (view == kViewGrid) {
        color = HashGridDebugColoredHash(surface.positionWorld, surface.normalWorld, cache.hashGridParameters);
    } else if (view == kViewRadiance) {
        float3 radiance;
        if (SharcGetCachedRadiance(cache, surface, radiance, false)) {
            color = radiance * params.exposureTexture[uint2(0)];
        }
    }
    params.gbuffer.writeOverlay(texel, color);
}
