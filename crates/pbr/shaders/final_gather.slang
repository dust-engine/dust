import pbr_common;
import sky_atmosphere;

struct [raypayload] GatherPayload {
    float dummy : read(caller) : write(caller, miss);
}

[vk::push_constant] uniform uint frameIndex;

[shader("raygeneration")]
void finalGatherRayGenShader() {
    uint2 pixel = DispatchRaysIndex().xy;

    float depth = params.gbuffer.depthTexture[pixel];
    if (depth == 0.0) {
        return; // Sky or UI pixel
    }

    // Reconstruct world position
    float3 worldPos = params.pixelPositionWorldSpace(DispatchRaysIndex().xy, DispatchRaysDimensions().xy);
    float3 N = params.gbuffer.normalTexture[pixel].xyz;

    // Sample blue noise cosine-weighted hemisphere direction
    uint2 texSize;
    uint _numLayers;
    params.blueNoiseCosine.GetDimensions(texSize.x, texSize.y, _numLayers);
    uint2 noiseCoord = pixel % texSize;
    float3 tangentDir = params.blueNoiseCosine.Load(int4(noiseCoord, frameIndex % 64, 0)).xyz * 2.0 - 1.0;

    // Build TBN from surface normal
    float3 T, B;
    if (abs(N.y) < 0.999) {
        T = normalize(cross(N, float3(0, 1, 0)));
    } else {
        T = normalize(cross(N, float3(1, 0, 0)));
    }
    B = cross(N, T);

    float3 worldDir = normalize(T * tangentDir.x + B * tangentDir.y + N * tangentDir.z);

    // Trace hemisphere ray
    RayDesc ray;
    ray.Origin = worldPos + N * 0.01;
    ray.Direction = worldDir;
    ray.TMin = 0.0;
    ray.TMax = params.uniforms.far;

    GatherPayload payload;
    payload.dummy = 0.0;
    TraceRay(
        params.sceneBVH,
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_FORCE_OPAQUE,
        ~0,
        0, 1, 0,
        ray, payload
    );
}

[shader("miss")]
void finalGatherMissShader() {
    // Ray escaped to sky — compute sky radiance contribution.
    uint2 pixel = DispatchRaysIndex().xy;
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

    // Check ground intersection
    float3 earthO = float3(0.0, 0.0, 0.0);
    bool intersectsGround = raySphereIntersectNearest(worldPos, worldDir, earthO, params.sky.atmosphereParams.bottom_radius) >= 0.0;

    float3 skyColor = float3(0.0);
    if (!intersectsGround && viewHeight < params.sky.atmosphereParams.top_radius) {
        float3 sideVector = normalize(cross(upVector, worldDir));
        float3 forwardVector = normalize(cross(sideVector, upVector));
        float2 lightOnPlane = float2(
            dot(params.sky.atmosphereParams.sun_direction, forwardVector),
            dot(params.sky.atmosphereParams.sun_direction, sideVector)
        );
        lightOnPlane = normalize(lightOnPlane);
        float lightViewCosAngle = lightOnPlane.x;

        float2 skyUV;
        skyViewParamsToUv(params.sky.atmosphereParams, false, viewZenithCosAngle, lightViewCosAngle, viewHeight, skyUV);
        skyColor = params.sky.skyViewLUT.SampleLevel(params.sky.linearSampler, skyUV, 0).rgb;
    }
    // No sun disk — shadow pass handles direct sun; including it here would cause fireflies at 1 spp.

    // Cosine-weighted importance sampling of Lambertian BRDF:
    // L_indirect = albedo * (1/PI) * L_i * cos(theta) / pdf = albedo * L_i
    float2 uv = (float2(pixel) + 0.5) / float2(DispatchRaysDimensions().xy);
    float3 albedo = params.gbuffer.albedoSRGB.SampleLevel(params.sky.linearSampler, uv, 0).rgb;

    float4 current = params.outputTexture[pixel];
    params.outputTexture[pixel] = float4(current.rgb + albedo * skyColor, current.a);
}
