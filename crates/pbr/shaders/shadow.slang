import pbr_common;
import sky_atmosphere;

struct [raypayload] ShadowPayload {
    float shadow : read(caller) : write(caller, miss);
}

[shader("raygeneration")]
void shadowRayGenShader() {
    uint2 pixel = DispatchRaysIndex().xy;
    uint missShaderIndex = params.gbuffer.albedoUint[pixel].a;

    // albedoUint.a is a per-pixel miss shader index written by the primary pass:
    // real hits store 0; sky/UI store 0xFF as a "no surface" sentinel. Skip the
    // sentinel pixels; the rest is passed straight to TraceRay's MissShaderIndex.
    if (missShaderIndex == 0xFF) {
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
        0, 1, missShaderIndex,
        ray, payload
    );
}

[shader("miss")]
void shadowMissShader() {
    uint2 pixel = DispatchRaysIndex().xy;
    float4 current = params.outputTexture[pixel];

    // Shadow ray missed — no blocker between surface and sun.
    // Compute direct sunlight contribution and add to render target.
    float3 samplePosFromCamera = params.pixelPositionWorldSpace(pixel, DispatchRaysDimensions().xy)
                               - params.uniforms.cameraOrigin();

    float3 transmittanceToSun = params.sky.sampleTransmittanceAt(samplePosFromCamera);

    // Sample albedo through sRGB view — hardware applies sRGB-to-linear conversion
    float2 uv = (float2(pixel) + 0.5) / float2(DispatchRaysDimensions().xy);
    float3 worldNormal = params.gbuffer.normalTexture[pixel].xyz;

    // Direct sun = (diffuse Lambertian + specular GGX) · E_incident.
    //   E_incident = solar_illuminance · transmittanceToSun · NdotL
    // The 1/π is the Lambertian normalization. specularGGX folds in the
    // 1/(4·NoV·NoL) microfacet denominator, so multiplying by NdotL leaves a
    // finite specular term. solar_illuminance is multiplied in explicitly
    // because transmittanceToSun is a unitless LUT lookup (see sky_view_lut).
    float3 sunDir = params.sky.atmosphereParams.sun_direction;
    float NdotL = max(0.0, dot(worldNormal, sunDir));
    float3 albedo = params.gbuffer.albedoSRGB[pixel].rgb;   // diffuse (metalness removed in the primary pass)
    float3 E_perp = params.sky.atmosphereParams.solar_illuminance * transmittanceToSun;

    // Primary-surface view vector + packed material for the specular lobe.
    float3 V = -params.uniforms.primaryRayDirWorldSpace(pixel, DispatchRaysDimensions().xy);
    float roughness = params.gbuffer.normalTexture[pixel].w;
    float3 F0 = params.gbuffer.specularAlbedo[pixel].rgb;
    float3 brdf = albedo / PI + specularGGX(worldNormal, V, sunDir, roughness, F0);

    params.outputTexture[pixel] = float4(current.rgb + brdf * E_perp * NdotL, current.a);
}
