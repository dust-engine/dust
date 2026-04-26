import pbr_common;
import sky_atmosphere;

struct [raypayload] ShadowPayload {
    float shadow : read(caller) : write(caller, miss);
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
    float3 worldPos = params.pixelPositionWorldSpace(DispatchRaysIndex().xy, DispatchRaysDimensions().xy);
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
    float depth = params.gbuffer.depthTexture[pixel];
    float3 samplePosFromCamera = params.uniforms.primaryRayDirWorldSpace(DispatchRaysIndex().xy, DispatchRaysDimensions().xy) * depth;

    float3 transmittanceToSun = params.sky.sampleTransmittanceAt(samplePosFromCamera);

    // Sample albedo through sRGB view — hardware applies sRGB-to-linear conversion
    float2 uv = (float2(pixel) + 0.5) / float2(DispatchRaysDimensions().xy);
    float3 worldNormal = params.gbuffer.normalTexture[pixel].xyz;

    // Direct sun contribution via Lambertian BRDF:
    //   L_out [cd/m²] = (albedo / π [sr⁻¹]) * E_incident [lux]
    //   E_incident   = solar_illuminance * transmittanceToSun * NdotL
    // The 1/π is the Lambertian BRDF normalization; without it the surface
    // would be ~π× too bright. solar_illuminance is multiplied in explicitly
    // because transmittanceToSun is a unitless LUT lookup (see sky_view_lut).
    float NdotL = max(0.0, dot(worldNormal, params.sky.atmosphereParams.sun_direction));
    float3 sunContribution = 1.0 / PI // albedo is demodulated
                           * params.sky.atmosphereParams.solar_illuminance
                           * transmittanceToSun
                           * NdotL;

    params.outputTexture[pixel] = float4(current.rgb + sunContribution, current.a);
}
