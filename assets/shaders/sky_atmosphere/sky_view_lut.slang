// Sky View LUT Compute Shader
// Generates a per-frame 2D lookup table of sky radiance
// Parameterized by view zenith angle and light-view angle
// Output: 192x108 R16G16B16A16_SFLOAT texture

#include "atmosphere_common.slang"

[[vk::binding(0, 0)]] ConstantBuffer<AtmosphereParams> params;
[[vk::binding(1, 0)]] Texture2D<float4> transmittanceLUT;
[[vk::binding(2, 0)]] Texture2D<float4> multiScatteringLUT;
[[vk::binding(3, 0)]] SamplerState linearSampler;
[format("rgba16f")]
[[vk::binding(4, 0)]] RWTexture2D<float4> skyViewLUT;

// Sample transmittance from LUT
float3 sampleTransmittance(AtmosphereParams atm, float viewHeight, float viewZenithCosAngle) {
    float2 uv;
    transmittanceParamsToUv(atm, viewHeight, viewZenithCosAngle, uv);
    return transmittanceLUT.SampleLevel(linearSampler, uv, 0).rgb;
}

// Sample multi-scattering from LUT
float3 sampleMultiScattering(AtmosphereParams atm, float3 worldPos, float viewZenithCosAngle) {
    float2 uv = saturate(float2(
        viewZenithCosAngle * 0.5 + 0.5,
        (length(worldPos) - atm.bottom_radius) / (atm.top_radius - atm.bottom_radius)
    ));
    uv = float2(fromUnitToSubUvs(uv.x, float(MULTI_SCATTERING_RES)),
                fromUnitToSubUvs(uv.y, float(MULTI_SCATTERING_RES)));
    return multiScatteringLUT.SampleLevel(linearSampler, uv, 0).rgb;
}

// Integrate scattered luminance along a ray
float3 integrateScatteredLuminance(
    float3 worldPos,
    float3 worldDir,
    float3 sunDir,
    AtmosphereParams atm
) {
    // Ray-atmosphere intersection
    float3 earthO = float3(0.0, 0.0, 0.0);
    float tBottom = raySphereIntersectNearest(worldPos, worldDir, earthO, atm.bottom_radius);
    float tTop = raySphereIntersectNearest(worldPos, worldDir, earthO, atm.top_radius);

    float tMax = 0.0;
    if (tBottom < 0.0) {
        if (tTop < 0.0) {
            return float3(0.0, 0.0, 0.0);
        }
        tMax = tTop;
    } else {
        tMax = tTop > 0.0 ? min(tTop, tBottom) : tBottom;
    }

    // Variable sample count based on ray length
    float sampleCount = lerp(4.0, 14.0, saturate(tMax * 0.01));
    float sampleCountFloor = floor(sampleCount);
    float tMaxFloor = tMax * sampleCountFloor / sampleCount;

    // Phase functions
    float cosTheta = dot(sunDir, worldDir);
    float miePhaseValue = hgPhase(atm.mie_phase_g, -cosTheta);
    float rayleighPhaseValue = rayleighPhase(cosTheta);

    // Ray march
    float3 L = float3(0.0, 0.0, 0.0);
    float3 throughput = float3(1.0, 1.0, 1.0);
    float t = 0.0;

    for (float s = 0.0; s < sampleCount; s += 1.0) {
        // Non-linear sample distribution
        float t0 = s / sampleCountFloor;
        float t1 = (s + 1.0) / sampleCountFloor;
        t0 = t0 * t0;
        t1 = t1 * t1;
        t0 = tMaxFloor * t0;
        t1 = (t1 > 1.0) ? tMax : tMaxFloor * t1;

        float dt = t1 - t0;
        t = t0 + dt * 0.3;

        float3 P = worldPos + t * worldDir;
        MediumSample medium = sampleMedium(P, atm);

        float3 sampleOpticalDepth = medium.extinction * dt;
        float3 stepTransmittance = exp(-sampleOpticalDepth);

        // Transmittance to sun
        float pHeight = length(P);
        float3 upVector = P / pHeight;
        float sunZenithCosAngle = dot(sunDir, upVector);
        float3 transmittanceToSun = sampleTransmittance(atm, pHeight, sunZenithCosAngle);

        // Earth shadow
        float3 earthCenter = earthO + PLANET_RADIUS_OFFSET * upVector;
        float tEarth = raySphereIntersectNearest(P, sunDir, earthCenter, atm.bottom_radius);
        float earthShadow = tEarth >= 0.0 ? 0.0 : 1.0;

        // Phase-weighted scattering
        float3 phaseTimesScattering = medium.scattering_mie * miePhaseValue +
                                       medium.scattering_ray * rayleighPhaseValue;

        // Multi-scattering contribution
        float3 multiScatteredLuminance = sampleMultiScattering(atm, P, sunZenithCosAngle);

        // Total in-scattered light
        float3 S = atm.solar_irradiance * (earthShadow * transmittanceToSun * phaseTimesScattering +
                                           multiScatteredLuminance * medium.scattering);

        // Analytical integration over segment
        float3 Sint = (S - S * stepTransmittance) / medium.extinction;
        L += throughput * Sint;
        throughput *= stepTransmittance;
    }

    return L;
}

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    int2 coord = int2(dispatchThreadID.xy);

    if (coord.x >= SKY_VIEW_WIDTH || coord.y >= SKY_VIEW_HEIGHT) {
        return;
    }

    float2 uv = (float2(coord) + 0.5) / float2(SKY_VIEW_WIDTH, SKY_VIEW_HEIGHT);

    // Camera setup - Bevy uses Y-up, so add bottom_radius to Y
    float3 worldPos = params.camera_position + float3(0.0, params.bottom_radius, 0.0);
    float viewHeight = length(worldPos);

    // Convert UV to view and light angles
    float viewZenithCosAngle;
    float lightViewCosAngle;
    uvToSkyViewParams(params, viewZenithCosAngle, lightViewCosAngle, viewHeight, uv);

    // Reconstruct sun direction relative to view
    float3 upVector = worldPos / viewHeight;
    float sunZenithCosAngle = dot(upVector, params.sun_direction);
    float3 sunDir = normalize(float3(sqrt(1.0 - sunZenithCosAngle * sunZenithCosAngle), 0.0, sunZenithCosAngle));

    // Construct view direction in canonical space
    worldPos = float3(0.0, 0.0, viewHeight);
    float viewZenithSinAngle = sqrt(1.0 - viewZenithCosAngle * viewZenithCosAngle);
    float3 worldDir = float3(
        viewZenithSinAngle * lightViewCosAngle,
        viewZenithSinAngle * sqrt(1.0 - lightViewCosAngle * lightViewCosAngle),
        viewZenithCosAngle
    );

    // Move to atmosphere boundary if outside
    if (!moveToTopAtmosphere(worldPos, worldDir, params.top_radius)) {
        skyViewLUT[coord] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    float3 L = integrateScatteredLuminance(worldPos, worldDir, sunDir, params);

    skyViewLUT[coord] = float4(L, 1.0);
}
