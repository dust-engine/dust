import pbr_common;
import sky_atmosphere;

struct [raypayload] RayPayload {
    float color : read(caller) : write(caller, closesthit, miss);
}

[shader("raygeneration")]
void rayGenShader() {
    // Check swapchain for egui occlusion
    float4 originalColor = params.gbuffer.occlusionTexture[DispatchRaysIndex().xy];
    if (originalColor.a == 1.0) {
        // Solid UI pixel. Write transparent to HDR so tonemap skips it.
        params.outputTexture[DispatchRaysIndex().xy] = float4(0, 0, 0, 0);
        params.gbuffer.albedoLinear[DispatchRaysIndex().xy] = float4(0, 0, 0, 0);
        params.gbuffer.normalTexture[DispatchRaysIndex().xy] = float4(0, 0, 0, 0);
        params.gbuffer.depthTexture[DispatchRaysIndex().xy] = 0.0;
        // No surface info under UI — reproject as if at infinity so DLSS sees
        // motion consistent with camera rotation rather than a stationary mask.
        float3 worldDir = params.uniforms.primaryRayDirWorldSpace(DispatchRaysIndex().xy, DispatchRaysDimensions().xy);
        float2 prevPixel = params.uniforms.prevPixelFromWorldDir(worldDir, DispatchRaysDimensions().xy);
        // Subtract jitter so currPixel is the *un-jittered* projection of the
        // sample we cast (the jittered ray sampled the world at sub-pixel
        // p - jitter in unjittered NDC). DLSS expects MV in unjittered grid.
        float2 currPixel = float2(DispatchRaysIndex().xy) - float2(params.uniforms.jitter_x, params.uniforms.jitter_y);
        params.gbuffer.motionVectorTexture[DispatchRaysIndex().xy] = prevPixel - currPixel;
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
    params.gbuffer.albedoLinear[DispatchRaysIndex().xy] = float4(0, 0, 0, 0);
    params.gbuffer.normalTexture[DispatchRaysIndex().xy] = float4(0, 0, 0, 0);
    params.gbuffer.depthTexture[DispatchRaysIndex().xy] = 0.0;
    // Sky is at infinity — reproject the view direction through the previous
    // camera (rotation only) so DLSS sees motion from camera turn/FOV change.
    float2 prevPixel = params.uniforms.prevPixelFromWorldDir(worldDir, DispatchRaysDimensions().xy);
    // See pbr.slang ray-gen for the un-jittering rationale.
    float2 currPixel = float2(DispatchRaysIndex().xy) - float2(params.uniforms.jitter_x, params.uniforms.jitter_y);
    params.gbuffer.motionVectorTexture[DispatchRaysIndex().xy] = prevPixel - currPixel;
}
