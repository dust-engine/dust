// Transmittance LUT Compute Shader
// Computes optical depth (transmittance) from any point to the top of the atmosphere
// Output: 256x64 R16G16B16A16_SFLOAT texture

#include "atmosphere_common.slang"

[[vk::binding(0, 0)]] ConstantBuffer<AtmosphereParams> params;
[format("rgba16f")]
[[vk::binding(1, 0)]] RWTexture2D<float4> transmittanceLUT;

// Integrate optical depth along a ray from worldPos in worldDir direction
float3 integrateOpticalDepth(float3 worldPos, float3 worldDir, AtmosphereParams atm) {
    // Compute ray intersection with atmosphere boundary
    float3 earthO = float3(0.0, 0.0, 0.0);
    float tMax = raySphereIntersectNearest(worldPos, worldDir, earthO, atm.top_radius);

    if (tMax < 0.0) {
        return float3(0.0, 0.0, 0.0);
    }

    // Also check ground intersection
    float tGround = raySphereIntersectNearest(worldPos, worldDir, earthO, atm.bottom_radius);
    if (tGround > 0.0) {
        tMax = min(tMax, tGround);
    }

    // Ray march and integrate optical depth
    const int SAMPLE_COUNT = 40;
    float dt = tMax / float(SAMPLE_COUNT);

    float3 opticalDepth = float3(0.0, 0.0, 0.0);
    float t = 0.0;

    for (int i = 0; i < SAMPLE_COUNT; i++) {
        float newT = tMax * (float(i) + 0.3) / float(SAMPLE_COUNT);
        dt = newT - t;
        t = newT;

        float3 P = worldPos + t * worldDir;
        MediumSample medium = sampleMedium(P, atm);
        opticalDepth += medium.extinction * dt;
    }

    return opticalDepth;
}

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    int2 coord = int2(dispatchThreadID.xy);

    if (coord.x >= TRANSMITTANCE_WIDTH || coord.y >= TRANSMITTANCE_HEIGHT) {
        return;
    }

    // Convert pixel coordinate to UV
    float2 uv = (float2(coord) + 0.5) / float2(TRANSMITTANCE_WIDTH, TRANSMITTANCE_HEIGHT);

    // Convert UV to physical parameters
    float viewHeight;
    float viewZenithCosAngle;
    uvToTransmittanceParams(params, viewHeight, viewZenithCosAngle, uv);

    // Construct world position and direction
    float3 worldPos = float3(0.0, 0.0, viewHeight);
    float viewZenithSinAngle = sqrt(1.0 - viewZenithCosAngle * viewZenithCosAngle);
    float3 worldDir = float3(viewZenithSinAngle, 0.0, viewZenithCosAngle);

    // Compute optical depth
    float3 opticalDepth = integrateOpticalDepth(worldPos, worldDir, params);

    // Convert to transmittance
    float3 transmittance = exp(-opticalDepth);

    transmittanceLUT[coord] = float4(transmittance, 1.0);
}
