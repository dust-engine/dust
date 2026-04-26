import pbr_common;
import sky_atmosphere;

struct [raypayload] ShadowPayload {
    float shadow : read(caller) : write(caller, miss);
}

float3 cameraDir(uint2 position, uint2 screenSize, float tan_half_fov, matrix<float, 3, 3> rotation) {
    float2 pixelNDC = (float2(position) + 0.5) / float2(screenSize);
    float2 pixelCamera = 2 * pixelNDC - 1;
    pixelCamera.y *= -1;
    pixelCamera.x *= float(screenSize.x) / float(screenSize.y);
    pixelCamera *= tan_half_fov;
    return mul(rotation, float3(pixelCamera, -1));
}

[shader("raygeneration")]
void shadowRayGenShader() {
    uint2 pixel = DispatchRaysIndex().xy;

    float depth = params.gbuffer.depthTexture[pixel];

    if (depth == 0.0) {
        // No hit (sky or UI)
        return;
    }

    // Reconstruct world position from camera ray + hit distance
    float3 dir = cameraDir(
        pixel,
        DispatchRaysDimensions().xy,
        params.uniforms.tan_half_fov,
        float3x3(params.uniforms.model[0].xyz, params.uniforms.model[1].xyz, params.uniforms.model[2].xyz)
    );
    float3 origin = float3(params.uniforms.model[0].w, params.uniforms.model[1].w, params.uniforms.model[2].w);
    float3 worldPos = origin + dir * depth;
    float3 worldNormal = params.gbuffer.normalTexture[pixel].xyz;

    RayDesc ray;
    ray.Origin = worldPos + worldNormal * 0.01; // Offset along normal to avoid self-intersection
    ray.Direction = params.sky.atmosphereParams.sun_direction;
    ray.TMin = 0.0;
    ray.TMax = params.uniforms.far;

    ShadowPayload payload;
    payload.shadow = 0.0;
    TraceRay(
        params.sceneBVH,
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_FORCE_OPAQUE,
        ~0,
        0, 1, 0,
        ray, payload
    );
}

[shader("miss")]
void shadowMissShader() {
    uint2 pixel = DispatchRaysIndex().xy;
    float4 current = params.outputTexture[pixel];

    // Shadow ray missed — no blocker between surface and sun.
    // Compute direct sunlight contribution and add to render target.

    // Reconstruct surface position in atmosphere frame (planet-centered, +Y up,
    // bottom_radius offset) to sample transmittance from surface to sun — matches
    // Hillaire EGSR2020 ground shading and pbr.slang:88-90.
    float depth = params.gbuffer.depthTexture[pixel];
    float3 dir = cameraDir(
        pixel,
        DispatchRaysDimensions().xy,
        params.uniforms.tan_half_fov,
        float3x3(params.uniforms.model[0].xyz, params.uniforms.model[1].xyz, params.uniforms.model[2].xyz)
    );
    float3 surfAtmPos = params.sky.atmosphereParams.camera_position
        + float3(0.0, params.sky.atmosphereParams.bottom_radius, 0.0)
        + dir * depth;
    float viewHeight = max(length(surfAtmPos), params.sky.atmosphereParams.bottom_radius + 0.001);
    float3 upVector = surfAtmPos / viewHeight;
    float sunZenithCosAngle = dot(upVector, params.sky.atmosphereParams.sun_direction);
    float3 transmittanceToSun = sampleTransmittance(
        params.sky.atmosphereParams, viewHeight, sunZenithCosAngle,
        params.sky.transmittanceLUT, params.sky.linearSampler
    );

    // Sample albedo through sRGB view — hardware applies sRGB-to-linear conversion
    float2 uv = (float2(pixel) + 0.5) / float2(DispatchRaysDimensions().xy);
    float3 albedo = params.gbuffer.albedoSRGB.SampleLevel(params.sky.linearSampler, uv, 0).rgb;
    float3 worldNormal = params.gbuffer.normalTexture[pixel].xyz;

    float NdotL = max(0.0, dot(worldNormal, params.sky.atmosphereParams.sun_direction));
    float3 sunContribution = transmittanceToSun * albedo * params.sky.atmosphereParams.solar_irradiance * NdotL / PI;

    params.outputTexture[pixel] = float4(current.rgb + sunContribution, current.a);
}
