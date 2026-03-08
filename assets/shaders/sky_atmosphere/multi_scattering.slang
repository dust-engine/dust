// Multi-Scattering LUT Compute Shader
// Computes the multiple scattering contribution using the technique from EGSR 2020
// Uses shared memory reduction to integrate over the sphere
// Output: 32x32 R16G16B16A16_SFLOAT texture

#include "atmosphere_common.slang"

[[vk::binding(0, 0)]] ConstantBuffer<AtmosphereParams> params;
[[vk::binding(1, 0)]] Texture2D<float4> transmittanceLUT;
[[vk::binding(2, 0)]] SamplerState linearSampler;
[format("rgba16f")]
[[vk::binding(3, 0)]] RWTexture2D<float4> multiScatteringLUT;

// Shared memory for parallel reduction
groupshared float3 sharedMultiScatAs1[64];
groupshared float3 sharedL[64];

// Sample transmittance from LUT
float3 sampleTransmittance(AtmosphereParams atm, float viewHeight, float viewZenithCosAngle) {
    float2 uv;
    transmittanceParamsToUv(atm, viewHeight, viewZenithCosAngle, uv);
    return transmittanceLUT.SampleLevel(linearSampler, uv, 0).rgb;
}

// Single scattering result for multi-scattering computation
struct SingleScatteringResult {
    float3 L;               // Scattered luminance
    float3 multiScatAs1;    // Contribution to multi-scattering
};

// Integrate single scattering for multi-scattering LUT generation
SingleScatteringResult integrateScatteredLuminance(
    float3 worldPos,
    float3 worldDir,
    float3 sunDir,
    AtmosphereParams atm,
    bool ground
) {
    SingleScatteringResult result;
    result.L = float3(0.0, 0.0, 0.0);
    result.multiScatAs1 = float3(0.0, 0.0, 0.0);

    // Ray march parameters
    float3 earthO = float3(0.0, 0.0, 0.0);
    float tBottom = raySphereIntersectNearest(worldPos, worldDir, earthO, atm.bottom_radius);
    float tTop = raySphereIntersectNearest(worldPos, worldDir, earthO, atm.top_radius);

    float tMax = 0.0;
    if (tBottom < 0.0) {
        if (tTop < 0.0) {
            return result;
        }
        tMax = tTop;
    } else {
        tMax = tTop > 0.0 ? min(tTop, tBottom) : tBottom;
    }

    const int SAMPLE_COUNT = 20;
    float dt = tMax / float(SAMPLE_COUNT);

    // Phase functions
    float cosTheta = dot(sunDir, worldDir);
    float miePhaseValue = hgPhase(atm.mie_phase_g, -cosTheta);
    float rayleighPhaseValue = rayleighPhase(cosTheta);

    // Ray march
    float3 throughput = float3(1.0, 1.0, 1.0);
    float t = 0.0;

    for (int i = 0; i < SAMPLE_COUNT; i++) {
        float newT = tMax * (float(i) + 0.3) / float(SAMPLE_COUNT);
        dt = newT - t;
        t = newT;

        float3 P = worldPos + t * worldDir;
        MediumSample medium = sampleMedium(P, atm);

        float3 sampleOpticalDepth = medium.extinction * dt;
        float3 stepTransmittance = exp(-sampleOpticalDepth);

        // Get transmittance to sun
        float pHeight = length(P);
        float3 upVector = P / pHeight;
        float sunZenithCosAngle = dot(sunDir, upVector);
        float3 transmittanceToSun = sampleTransmittance(atm, pHeight, sunZenithCosAngle);

        // Earth shadow
        float3 earthCenter = earthO + PLANET_RADIUS_OFFSET * upVector;
        float tEarth = raySphereIntersectNearest(P, sunDir, earthCenter, atm.bottom_radius);
        float earthShadow = tEarth >= 0.0 ? 0.0 : 1.0;

        // Scattering with uniform phase for multi-scattering integration
        float phase = uniformPhase();
        float3 scatteringNoPhase = medium.scattering * 1.0;

        // Multi-scattering contribution (Equation 7 from paper)
        float3 multiScatInt = (scatteringNoPhase - scatteringNoPhase * stepTransmittance) / medium.extinction;
        result.multiScatAs1 += throughput * multiScatInt;

        // In-scattered luminance from sun
        float3 S = earthShadow * transmittanceToSun * medium.scattering * phase;
        float3 Sint = (S - S * stepTransmittance) / medium.extinction;
        result.L += throughput * Sint;

        throughput *= stepTransmittance;
    }

    // Ground contribution
    if (ground && tMax == tBottom && tBottom > 0.0) {
        float3 P = worldPos + tBottom * worldDir;
        float pHeight = length(P);
        float3 upVector = P / pHeight;
        float sunZenithCosAngle = dot(sunDir, upVector);
        float3 transmittanceToSun = sampleTransmittance(atm, pHeight, sunZenithCosAngle);

        float NdotL = saturate(dot(upVector, sunDir));
        result.L += transmittanceToSun * throughput * NdotL * atm.ground_albedo / PI;
    }

    return result;
}

[numthreads(1, 1, 64)]
void main(uint3 threadId : SV_DispatchThreadID) {
    float2 pixPos = float2(threadId.xy) + 0.5;
    float2 uv = pixPos / float(MULTI_SCATTERING_RES);

    uv = float2(fromSubUvsToUnit(uv.x, float(MULTI_SCATTERING_RES)),
                fromSubUvsToUnit(uv.y, float(MULTI_SCATTERING_RES)));

    // Convert UV to physical parameters
    float cosSunZenithAngle = uv.x * 2.0 - 1.0;
    float3 sunDir = float3(0.0, sqrt(saturate(1.0 - cosSunZenithAngle * cosSunZenithAngle)), cosSunZenithAngle);

    float viewHeight = params.bottom_radius + saturate(uv.y + PLANET_RADIUS_OFFSET) *
        (params.top_radius - params.bottom_radius - PLANET_RADIUS_OFFSET);

    float3 worldPos = float3(0.0, 0.0, viewHeight);

    // Sphere integration using 8x8 uniform samples
    const float SQRT_SAMPLE_COUNT = 8.0;
    float i = 0.5 + float(threadId.z / 8);
    float j = 0.5 + float(threadId.z % 8);

    float randA = i / SQRT_SAMPLE_COUNT;
    float randB = j / SQRT_SAMPLE_COUNT;

    // Uniform sphere sampling
    float theta = 2.0 * PI * randA;
    float phi = acos(1.0 - 2.0 * randB);

    float cosPhi = cos(phi);
    float sinPhi = sin(phi);
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);

    float3 worldDir = float3(cosTheta * sinPhi, sinTheta * sinPhi, cosPhi);

    // Integrate scattering for this direction
    SingleScatteringResult result = integrateScatteredLuminance(
        worldPos, worldDir, sunDir, params, true);

    float sphereSolidAngle = 4.0 * PI;
    sharedMultiScatAs1[threadId.z] = result.multiScatAs1 * sphereSolidAngle / (SQRT_SAMPLE_COUNT * SQRT_SAMPLE_COUNT);
    sharedL[threadId.z] = result.L * sphereSolidAngle / (SQRT_SAMPLE_COUNT * SQRT_SAMPLE_COUNT);

    GroupMemoryBarrierWithGroupSync();

    // Parallel reduction
    if (threadId.z < 32) {
        sharedMultiScatAs1[threadId.z] += sharedMultiScatAs1[threadId.z + 32];
        sharedL[threadId.z] += sharedL[threadId.z + 32];
    }
    GroupMemoryBarrierWithGroupSync();

    if (threadId.z < 16) {
        sharedMultiScatAs1[threadId.z] += sharedMultiScatAs1[threadId.z + 16];
        sharedL[threadId.z] += sharedL[threadId.z + 16];
    }
    GroupMemoryBarrierWithGroupSync();

    if (threadId.z < 8) {
        sharedMultiScatAs1[threadId.z] += sharedMultiScatAs1[threadId.z + 8];
        sharedL[threadId.z] += sharedL[threadId.z + 8];
    }
    GroupMemoryBarrierWithGroupSync();

    if (threadId.z < 4) {
        sharedMultiScatAs1[threadId.z] += sharedMultiScatAs1[threadId.z + 4];
        sharedL[threadId.z] += sharedL[threadId.z + 4];
    }
    GroupMemoryBarrierWithGroupSync();

    if (threadId.z < 2) {
        sharedMultiScatAs1[threadId.z] += sharedMultiScatAs1[threadId.z + 2];
        sharedL[threadId.z] += sharedL[threadId.z + 2];
    }
    GroupMemoryBarrierWithGroupSync();

    if (threadId.z < 1) {
        sharedMultiScatAs1[threadId.z] += sharedMultiScatAs1[threadId.z + 1];
        sharedL[threadId.z] += sharedL[threadId.z + 1];
    }
    GroupMemoryBarrierWithGroupSync();

    if (threadId.z > 0) {
        return;
    }

    // Final computation (Equations 7 and 10 from paper)
    float isotropicPhase = uniformPhase();
    float3 multiScatAs1 = sharedMultiScatAs1[0] * isotropicPhase;
    float3 inScatteredLuminance = sharedL[0] * isotropicPhase;

    // Geometric series: sum = 1 / (1 - r) for infinite scattering orders
    float3 r = multiScatAs1;
    float3 sumAllScattering = 1.0 / (1.0 - r);
    float3 L = inScatteredLuminance * sumAllScattering;

    multiScatteringLUT[threadId.xy] = float4(L, 1.0);
}
