import pbr_common;
import sky_atmosphere;

struct [raypayload] RayPayload {
    float color : read(caller) : write(caller, closesthit, miss);
}

[shader("raygeneration")]
void rayGenShader() {
    // Check the egui overlay for occlusion. The overlay (occlusionTexture) is
    // at display resolution, but this pass runs at render resolution, which is
    // lower whenever DLSS upscaling is active. Map this render-res pixel onto
    // the overlay via nearest sampling — `(pixel + 0.5) / renderExtent` UVs
    // scaled by the overlay's own dimensions — instead of indexing it directly.
    uint2 overlayDim;
    params.gbuffer.occlusionTexture.GetDimensions(overlayDim.x, overlayDim.y);
    uint2 overlayPixel = uint2(
        (float2(DispatchRaysIndex().xy) + 0.5) * float2(overlayDim) / float2(DispatchRaysDimensions().xy));
    float4 originalColor = params.gbuffer.occlusionTexture[overlayPixel];
    if (originalColor.a == 1.0) {
        // Solid UI pixel. Write transparent to HDR so tonemap skips it.
        params.outputTexture[DispatchRaysIndex().xy] = float4(0, 0, 0, 0);
        params.gbuffer.albedoUint[DispatchRaysIndex().xy] = uint4(0, 0, 0, 0xFF);
        params.gbuffer.normalTexture[DispatchRaysIndex().xy] = float4(0, 0, 0, 0);
        params.gbuffer.specularAlbedo[DispatchRaysIndex().xy] = float4(0, 0, 0, 0);
        // UI occludes the world: treat it as a screen-locked surface stuck to
        // the near plane. depth = 1 in reverse-infinite-Z (closest), and MV = 0
        // because UI is fixed in screen space — last frame the same screen
        // pixel was the same UI element, not a rotated world direction.
        params.gbuffer.depthTexture[DispatchRaysIndex().xy] = 1.0;
        params.gbuffer.motionVectorTexture[DispatchRaysIndex().xy] = float2(0, 0);
        return;
    } else {
        params.outputTexture[DispatchRaysIndex().xy] = float4(0.0, 0.0, 0.0, 1.0);
    }
    RayDesc ray;
    ray.Direction = params.uniforms.primaryRayDirWorldSpace(DispatchRaysIndex().xy, DispatchRaysDimensions().xy);
    ray.Origin = params.uniforms.cameraOrigin();
    ray.TMin = params.uniforms.near;
    ray.TMax = params.uniforms.far;

    RayPayload payload;
    payload.color = 0.0;

    TraceRay(params.sceneBVH, RAY_FLAG_NONE, ~0, 0, 1, 0, ray, payload);
}

[shader("miss")]
void missShader() {
    float3 worldDir = WorldRayDirection();

    // Photometric pipeline (units: cd/m²). sampleScatteredTransfer returns a
    // unitless transfer factor (LUT is baked with solar_illuminance = 1), so we
    // must multiply by solar_illuminance [lux] to recover luminance [cd/m²].
    // getSunLuminance is already a terminal luminance value — no multiply.
    float3 finalColor = float3(0.0, 0.0, 0.0);
    finalColor += params.sky.atmosphereParams.solar_illuminance * params.sky.sampleScatteredTransfer(worldDir);
    finalColor += params.sky.getSunLuminance(worldDir);

    // Write linear HDR (tonemapping happens in dedicated compute pass)
    params.outputTexture[DispatchRaysIndex().xy] = float4(finalColor, 1.0);
    params.gbuffer.albedoUint[DispatchRaysIndex().xy] = uint4(0, 0, 0, 0xFF);
    params.gbuffer.normalTexture[DispatchRaysIndex().xy] = float4(0, 0, 0, 0);
    params.gbuffer.specularAlbedo[DispatchRaysIndex().xy] = float4(0, 0, 0, 0);
    params.gbuffer.depthTexture[DispatchRaysIndex().xy] = 0.0;
    // Sky is at infinity — reproject the view direction through the previous
    // camera (rotation only) so DLSS sees motion from camera turn/FOV change.
    float2 prevPixel = params.uniforms.prevPixelFromWorldDir(worldDir, DispatchRaysDimensions().xy);
    // See pbr.slang ray-gen for the un-jittering rationale.
    float2 currPixel = float2(DispatchRaysIndex().xy) - float2(params.uniforms.jitter_x, params.uniforms.jitter_y);
    params.gbuffer.motionVectorTexture[DispatchRaysIndex().xy] = prevPixel - currPixel;
}
