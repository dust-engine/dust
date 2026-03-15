#include "../sky_atmosphere/atmosphere_common.slang"

uniform RaytracingAccelerationStructure sceneBVH;
uniform RWTexture2D outputTexture;
ConstantBuffer<Uniforms> uniforms;

// Sky atmosphere bindings
[[vk::binding(3, 0)]] ConstantBuffer<AtmosphereParams> atmosphereParams;
[[vk::binding(4, 0)]] Texture2D<float4> transmittanceLUT;
[[vk::binding(5, 0)]] Texture2D<float4> skyViewLUT;
[[vk::binding(6, 0)]] SamplerState linearSampler;
[[vk::binding(7, 0)]] RWTexture2D<float4> occlusionTexture;

struct Uniforms {
    matrix<float, 3, 4> model;
    float tan_half_fov;
    float far;
    float near;
    float _padding;
}

struct [raypayload] RayPayload {
    float color : read(caller) : write(caller, closesthit, miss);
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
void rayGenShader() {
    // Check swapchain for egui occlusion
    float4 originalColor = occlusionTexture[DispatchRaysIndex().xy];
    if (originalColor.a == 1.0) {
        // Solid UI pixel. Write transparent to HDR so tonemap skips it.
        outputTexture[DispatchRaysIndex().xy] = float4(0, 0, 0, 0);
        return;
    }
    RayDesc ray;
    ray.Direction = cameraDir(
        DispatchRaysIndex().xy,
        DispatchRaysDimensions().xy,
        uniforms.tan_half_fov,
        float3x3(uniforms.model[0].xyz, uniforms.model[1].xyz, uniforms.model[2].xyz)
    );
    ray.Origin = float3(uniforms.model[0].w, uniforms.model[1].w, uniforms.model[2].w);
    ray.TMin = uniforms.near;
    ray.TMax = uniforms.far;

    RayPayload payload;
    payload.color = 0.0;

    TraceRay(sceneBVH, RAY_FLAG_NONE, ~0, 0, 1, 0, ray, payload);
}

// Sample transmittance from LUT
float3 sampleTransmittance(AtmosphereParams atm, float viewHeight, float viewZenithCosAngle) {
    float2 uv;
    transmittanceParamsToUv(atm, viewHeight, viewZenithCosAngle, uv);
    return transmittanceLUT.SampleLevel(linearSampler, uv, 0).rgb;
}

[shader("miss")]
void missShader() {
    float3 worldDir = normalize(WorldRayDirection());

    // Camera in atmosphere space (planet center at origin)
    float3 worldPos = atmosphereParams.camera_position + float3(0.0, atmosphereParams.bottom_radius, 0.0);
    float viewHeight = length(worldPos);

    if (viewHeight < atmosphereParams.bottom_radius + 0.001) {
        viewHeight = atmosphereParams.bottom_radius + 0.001;
        worldPos = normalize(worldPos) * viewHeight;
    }

    float3 upVector = worldPos / viewHeight;
    float viewZenithCosAngle = dot(worldDir, upVector);

    // Check if ray intersects ground
    float3 earthO = float3(0.0, 0.0, 0.0);
    bool intersectsGround = raySphereIntersectNearest(worldPos, worldDir, earthO, atmosphereParams.bottom_radius) >= 0.0;

    float3 finalColor = float3(0.0, 0.0, 0.0);

    if (intersectsGround) {
        // Ground rendering
        float tGround = raySphereIntersectNearest(worldPos, worldDir, earthO, atmosphereParams.bottom_radius);
        float3 groundPos = worldPos + worldDir * tGround;
        float3 groundNormal = normalize(groundPos);
        float groundHeight = length(groundPos);

        // Transmittance from camera to ground
        float3 transmittanceToGround = sampleTransmittance(atmosphereParams, viewHeight, viewZenithCosAngle);

        // Transmittance from ground to sun
        float sunZenithCosAngle = dot(groundNormal, atmosphereParams.sun_direction);
        float3 transmittanceGroundToSun = sampleTransmittance(atmosphereParams, groundHeight, sunZenithCosAngle);

        // Ground is lit if sun is above horizon
        float NdotL = max(0.0, dot(groundNormal, atmosphereParams.sun_direction));

        // Simple checkerboard pattern for ground visualization
        float2 groundUV = float2(
            atan2(groundPos.x, groundPos.y),
            asin(groundPos.z / groundHeight)
        ) * 10000.0;
        float checker = abs(fmod(floor(groundUV.x) + floor(groundUV.y), 2.0));
        float3 groundColor = lerp(atmosphereParams.ground_albedo * 0.5, atmosphereParams.ground_albedo, checker);

        // Ground radiance
        float3 groundRadiance = groundColor * atmosphereParams.solar_irradiance * transmittanceGroundToSun * NdotL / PI;
        finalColor = groundRadiance * transmittanceToGround;

        // Add in-scattered light from atmosphere between camera and ground
        if (viewHeight < atmosphereParams.top_radius) {
            float3 sideVector = normalize(cross(upVector, worldDir));
            float3 forwardVector = normalize(cross(sideVector, upVector));
            float2 lightOnPlane = float2(dot(atmosphereParams.sun_direction, forwardVector), dot(atmosphereParams.sun_direction, sideVector));
            lightOnPlane = normalize(lightOnPlane);
            float lightViewCosAngle = lightOnPlane.x;

            float2 skyUV;
            skyViewParamsToUv(atmosphereParams, true, viewZenithCosAngle, lightViewCosAngle, viewHeight, skyUV);
            float3 inscatter = skyViewLUT.SampleLevel(linearSampler, skyUV, 0).rgb;

            float inscatterStrength = 1.0 - exp(-tGround * 0.0001);
            finalColor = finalColor * (1.0 - inscatterStrength) + inscatter * inscatterStrength;
        }
    } else {
        // Sky rendering using Sky View LUT
        if (viewHeight < atmosphereParams.top_radius) {
            float3 sideVector = normalize(cross(upVector, worldDir));
            float3 forwardVector = normalize(cross(sideVector, upVector));
            float2 lightOnPlane = float2(dot(atmosphereParams.sun_direction, forwardVector), dot(atmosphereParams.sun_direction, sideVector));
            lightOnPlane = normalize(lightOnPlane);
            float lightViewCosAngle = lightOnPlane.x;

            float2 skyUV;
            skyViewParamsToUv(atmosphereParams, false, viewZenithCosAngle, lightViewCosAngle, viewHeight, skyUV);

            finalColor = skyViewLUT.SampleLevel(linearSampler, skyUV, 0).rgb;
        }

        // Add sun disk
        finalColor += getSunLuminance(worldPos, worldDir, atmosphereParams.sun_direction, atmosphereParams.sun_angular_radius, atmosphereParams.bottom_radius);
    }

    // Write linear HDR (tonemapping happens in dedicated compute pass)
    outputTexture[DispatchRaysIndex().xy] = float4(finalColor, 1.0);
}
