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
        return;
    } else {
        params.outputTexture[DispatchRaysIndex().xy] = float4(0.0, 0.0, 0.0, 1.0);
    }
    RayDesc ray;
    ray.Direction = params.uniforms.cameraDir(DispatchRaysIndex().xy, DispatchRaysDimensions().xy);
    ray.Origin = params.uniforms.cameraOrigin();
    ray.TMin = params.uniforms.near;
    ray.TMax = params.uniforms.far;

    RayPayload payload;
    payload.color = 0.0;

    TraceRay(params.sceneBVH, RAY_FLAG_NONE, ~0, 0, 1, 0, ray, payload);
}

// Sample transmittance from LUT
float3 sampleTransmittance(SkyAtmosphereParams atm, float viewHeight, float viewZenithCosAngle) {
    float2 uv;
    transmittanceParamsToUv(atm.atmosphereParams, viewHeight, viewZenithCosAngle, uv);
    return  atm.transmittanceLUT.SampleLevel(atm.linearSampler, uv, 0).rgb;
}

[shader("miss")]
void missShader() {
    float3 worldDir = normalize(WorldRayDirection());

    // Camera in atmosphere space (planet center at origin)
    float3 worldPos = params.sky.atmosphereParams.camera_position + float3(0.0, params.sky.atmosphereParams.bottom_radius, 0.0);
    float viewHeight = length(worldPos);

    if (viewHeight < params.sky.atmosphereParams.bottom_radius + 0.001) {
        viewHeight = params.sky.atmosphereParams.bottom_radius + 0.001;
        worldPos = normalize(worldPos) * viewHeight;
    }

    float3 upVector = worldPos / viewHeight;
    float viewZenithCosAngle = dot(worldDir, upVector);

    // Check if ray intersects ground
    float3 earthO = float3(0.0, 0.0, 0.0);
    bool intersectsGround = raySphereIntersectNearest(worldPos, worldDir, earthO, params.sky.atmosphereParams.bottom_radius) >= 0.0;

    float3 finalColor = float3(0.0, 0.0, 0.0);

    if (intersectsGround) {
        // Ground rendering
        float tGround = raySphereIntersectNearest(worldPos, worldDir, earthO, params.sky.atmosphereParams.bottom_radius);
        float3 groundPos = worldPos + worldDir * tGround;
        float3 groundNormal = normalize(groundPos);
        float groundHeight = length(groundPos);

        // Transmittance from camera to ground
        float3 transmittanceToGround = sampleTransmittance(params.sky, viewHeight, viewZenithCosAngle);

        // Transmittance from ground to sun
        float sunZenithCosAngle = dot(groundNormal, params.sky.atmosphereParams.sun_direction);
        float3 transmittanceGroundToSun = sampleTransmittance(params.sky, groundHeight, sunZenithCosAngle);

        // Ground is lit if sun is above horizon
        float NdotL = max(0.0, dot(groundNormal, params.sky.atmosphereParams.sun_direction));

        // Simple checkerboard pattern for ground visualization
        float2 groundUV = float2(
            atan2(groundPos.x, groundPos.y),
            asin(groundPos.z / groundHeight)
        ) * 10000.0;
        float checker = abs(fmod(floor(groundUV.x) + floor(groundUV.y), 2.0));
        float3 groundColor = lerp(params.sky.atmosphereParams.ground_albedo * 0.5, params.sky.atmosphereParams.ground_albedo, checker);

        // Ground radiance
        float3 groundRadiance = groundColor * params.sky.atmosphereParams.solar_irradiance * transmittanceGroundToSun * NdotL / PI;
        finalColor = groundRadiance * transmittanceToGround;

        // Add in-scattered light from atmosphere between camera and ground
        if (viewHeight < params.sky.atmosphereParams.top_radius) {
            float3 sideVector = normalize(cross(upVector, worldDir));
            float3 forwardVector = normalize(cross(sideVector, upVector));
            float2 lightOnPlane = float2(dot(params.sky.atmosphereParams.sun_direction, forwardVector), dot(params.sky.atmosphereParams.sun_direction, sideVector));
            lightOnPlane = normalize(lightOnPlane);
            float lightViewCosAngle = lightOnPlane.x;

            float2 skyUV;
            skyViewParamsToUv(params.sky.atmosphereParams, true, viewZenithCosAngle, lightViewCosAngle, viewHeight, skyUV);
            float3 inscatter = params.sky.skyViewLUT.SampleLevel(params.sky.linearSampler, skyUV, 0).rgb;

            float inscatterStrength = 1.0 - exp(-tGround * 0.0001);
            finalColor = finalColor * (1.0 - inscatterStrength) + inscatter * inscatterStrength;
        }
    } else {
        // Sky rendering using Sky View LUT
        if (viewHeight < params.sky.atmosphereParams.top_radius) {
            float3 sideVector = normalize(cross(upVector, worldDir));
            float3 forwardVector = normalize(cross(sideVector, upVector));
            float2 lightOnPlane = float2(dot(params.sky.atmosphereParams.sun_direction, forwardVector), dot(params.sky.atmosphereParams.sun_direction, sideVector));
            lightOnPlane = normalize(lightOnPlane);
            float lightViewCosAngle = lightOnPlane.x;

            float2 skyUV;
            skyViewParamsToUv(params.sky.atmosphereParams, false, viewZenithCosAngle, lightViewCosAngle, viewHeight, skyUV);

            finalColor = params.sky.skyViewLUT.SampleLevel(params.sky.linearSampler, skyUV, 0).rgb;
        }

        // Add sun disk
        finalColor += getSunLuminance(worldPos, worldDir, params.sky.atmosphereParams.sun_direction, params.sky.atmosphereParams.sun_angular_radius, params.sky.atmosphereParams.bottom_radius);
    }

    // Write linear HDR (tonemapping happens in dedicated compute pass)
    params.outputTexture[DispatchRaysIndex().xy] = float4(finalColor, 1.0);
    params.gbuffer.albedoLinear[DispatchRaysIndex().xy] = float4(0, 0, 0, 0);
    params.gbuffer.normalTexture[DispatchRaysIndex().xy] = float4(0, 0, 0, 0);
    params.gbuffer.depthTexture[DispatchRaysIndex().xy] = 0.0;
}
