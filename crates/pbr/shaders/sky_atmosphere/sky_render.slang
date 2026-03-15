// Sky Render Shader - Final rendering pass
// Renders sky using Sky View LUT and a simple ground plane

#include "atmosphere_common.slang"

[[vk::binding(0, 0)]] ConstantBuffer<AtmosphereParams> params;
[[vk::binding(1, 0)]] Texture2D<float4> transmittanceLUT;
[[vk::binding(2, 0)]] Texture2D<float4> skyViewLUT;
[[vk::binding(3, 0)]] SamplerState linearSampler;

// Vertex shader output
struct VertexOutput {
    float4 position : SV_Position;
};

// Full-screen triangle vertex shader
[shader("vertex")]
VertexOutput vertexMain(uint vertexID : SV_VertexID) {
    VertexOutput output;

    // Generate full-screen triangle
    float2 uv = float2((vertexID << 1) & 2, vertexID & 2);
    output.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);

    return output;
}

// Sample transmittance from LUT
float3 sampleTransmittance(AtmosphereParams atm, float viewHeight, float viewZenithCosAngle) {
    float2 uv;
    transmittanceParamsToUv(atm, viewHeight, viewZenithCosAngle, uv);
    return transmittanceLUT.SampleLevel(linearSampler, uv, 0).rgb;
}

// Simple ACES tone mapping
float3 acesToneMap(float3 color) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return saturate((color * (a * color + b)) / (color * (c * color + d) + e));
}

// Fragment shader
[shader("fragment")]
float4 fragmentMain(float4 input: SV_Position) : SV_Target {
    float2 pixPos = input.xy;

    // Reconstruct world-space ray direction
    float2 clipXY = (pixPos / params.resolution) * 2.0 - 1.0;
    clipXY.y *= -1.0;

    float4 viewPos = mul(params.inv_proj_mat, float4(clipXY, 1.0, 1.0));
    float3 worldDir = normalize(mul((float3x3)params.inv_view_mat, viewPos.xyz / viewPos.w));

    // Camera in atmosphere space (planet center at origin)
    float3 worldPos = params.camera_position + float3(0.0, params.bottom_radius, 0.0);
    float viewHeight = length(worldPos);

    // Check if camera is below ground (shouldn't happen normally)
    if (viewHeight < params.bottom_radius + 0.001) {
        viewHeight = params.bottom_radius + 0.001;
        worldPos = normalize(worldPos) * viewHeight;
    }

    float3 upVector = worldPos / viewHeight;
    float viewZenithCosAngle = dot(worldDir, upVector);

    // Check if ray intersects ground
    float3 earthO = float3(0.0, 0.0, 0.0);
    bool intersectsGround = raySphereIntersectNearest(worldPos, worldDir, earthO, params.bottom_radius) >= 0.0;

    float3 finalColor = float3(0.0, 0.0, 0.0);

    if (intersectsGround) {
        // Ground rendering
        float tGround = raySphereIntersectNearest(worldPos, worldDir, earthO, params.bottom_radius);
        float3 groundPos = worldPos + worldDir * tGround;
        float3 groundNormal = normalize(groundPos);
        float groundHeight = length(groundPos);

        // Transmittance from camera to ground
        float3 transmittanceToGround = sampleTransmittance(params, viewHeight, viewZenithCosAngle);

        // Transmittance from ground to sun
        float sunZenithCosAngle = dot(groundNormal, params.sun_direction);
        float3 transmittanceGroundToSun = sampleTransmittance(params, groundHeight, sunZenithCosAngle);

        // Ground is lit if sun is above horizon
        float NdotL = max(0.0, dot(groundNormal, params.sun_direction));

        // Simple checkerboard pattern for ground visualization
        float2 groundUV = float2(
            atan2(groundPos.x, groundPos.y),
            asin(groundPos.z / groundHeight)
        ) * 10000.0;
        float checker = abs(fmod(floor(groundUV.x) + floor(groundUV.y), 2.0));
        float3 groundColor = lerp(params.ground_albedo * 0.5, params.ground_albedo, checker);

        // Ground radiance
        float3 groundRadiance = groundColor * params.solar_irradiance * transmittanceGroundToSun * NdotL / PI;
        finalColor = groundRadiance * transmittanceToGround;

        // Add in-scattered light from atmosphere between camera and ground
        // (simplified: use sky view LUT near horizon)
        if (viewHeight < params.top_radius) {
            float3 sideVector = normalize(cross(upVector, worldDir));
            float3 forwardVector = normalize(cross(sideVector, upVector));
            float2 lightOnPlane = float2(dot(params.sun_direction, forwardVector), dot(params.sun_direction, sideVector));
            lightOnPlane = normalize(lightOnPlane);
            float lightViewCosAngle = lightOnPlane.x;

            float2 skyUV;
            skyViewParamsToUv(params, true, viewZenithCosAngle, lightViewCosAngle, viewHeight, skyUV);
            float3 inscatter = skyViewLUT.SampleLevel(linearSampler, skyUV, 0).rgb;

            // Blend inscatter based on distance
            float inscatterStrength = 1.0 - exp(-tGround * 0.0001);
            finalColor = finalColor * (1.0 - inscatterStrength) + inscatter * inscatterStrength;
        }
    } else {
        // Sky rendering using Sky View LUT
        if (viewHeight < params.top_radius) {
            float3 sideVector = normalize(cross(upVector, worldDir));
            float3 forwardVector = normalize(cross(sideVector, upVector));
            float2 lightOnPlane = float2(dot(params.sun_direction, forwardVector), dot(params.sun_direction, sideVector));
            lightOnPlane = normalize(lightOnPlane);
            float lightViewCosAngle = lightOnPlane.x;

            float2 skyUV;
            skyViewParamsToUv(params, false, viewZenithCosAngle, lightViewCosAngle, viewHeight, skyUV);

            finalColor = skyViewLUT.SampleLevel(linearSampler, skyUV, 0).rgb;
        }

        // Add sun disk
        finalColor += getSunLuminance(worldPos, worldDir, params.sun_direction, params.sun_angular_radius, params.bottom_radius);
    }

    // Exposure and tone mapping
    float exposure = 10.0;
    finalColor = acesToneMap(finalColor * exposure);

    // Gamma correction (assuming sRGB output)
    finalColor = pow(finalColor, 1.0 / 2.2);

    return float4(finalColor, 1.0);
}
