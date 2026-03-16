#include "./sky_atmosphere/atmosphere_common.slang"

uniform RaytracingAccelerationStructure sceneBVH;
[[vk::binding(1, 0)]] uniform RWTexture2D<float4> outputTexture;
ConstantBuffer<Uniforms> uniforms;

[[vk::binding(3, 0)]] ConstantBuffer<AtmosphereParams> atmosphereParams;
[[vk::binding(6, 0)]] SamplerState linearSampler;
[[vk::binding(9, 0)]] uniform RWTexture2D<float4> normalTexture;
[[vk::binding(10, 0)]] uniform RWTexture2D<float> depthTexture;
[[vk::binding(11, 0)]] Texture2D<float4> albedoTexture;

struct Uniforms {
    matrix<float, 3, 4> model;
    float tan_half_fov;
    float far;
    float near;
    float _padding;
}

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

    float depth = depthTexture[pixel];

    if (depth == 0.0) {
        // No hit (sky or UI)
        return;
    }

    // Reconstruct world position from camera ray + hit distance
    float3 dir = cameraDir(
        pixel,
        DispatchRaysDimensions().xy,
        uniforms.tan_half_fov,
        float3x3(uniforms.model[0].xyz, uniforms.model[1].xyz, uniforms.model[2].xyz)
    );
    float3 origin = float3(uniforms.model[0].w, uniforms.model[1].w, uniforms.model[2].w);
    float3 worldPos = origin + dir * depth;
    float3 worldNormal = normalTexture[pixel].xyz;

    RayDesc ray;
    ray.Origin = worldPos + worldNormal * 0.01; // Offset along normal to avoid self-intersection
    ray.Direction = atmosphereParams.sun_direction;
    ray.TMin = 0.0;
    ray.TMax = uniforms.far;

    ShadowPayload payload;
    payload.shadow = 0.0;
    TraceRay(
        sceneBVH,
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_FORCE_OPAQUE,
        ~0,
        0, 1, 0,
        ray, payload
    );
}

[shader("miss")]
void shadowMissShader() {
    // Shadow ray missed — no blocker between surface and sun.
    // Compute direct sunlight contribution and add to render target.
    uint2 pixel = DispatchRaysIndex().xy;

    // Sample albedo through sRGB view — hardware applies sRGB-to-linear conversion
    float2 uv = (float2(pixel) + 0.5) / float2(DispatchRaysDimensions().xy);
    float3 albedo = albedoTexture.SampleLevel(linearSampler, uv, 0).rgb;
    float3 worldNormal = normalTexture[pixel].xyz;

    float NdotL = max(0.0, dot(worldNormal, atmosphereParams.sun_direction));
    float3 sunContribution = albedo * atmosphereParams.solar_irradiance * NdotL / PI;

    float4 current = outputTexture[pixel];
    outputTexture[pixel] = float4(current.rgb + sunContribution, current.a);
}
