import pbr_common;

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
    float3 worldPos = params.pixelPositionWorldSpace(pixel, DispatchRaysDimensions().xy);
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
    // Ray escaped to sky — compute sky luminance contribution.
    // No sun disk: shadow pass handles direct sun; including it here would cause
    // fireflies at 1 spp.
    uint2 pixel = DispatchRaysIndex().xy;
    float4 current = params.outputTexture[pixel];

    float3 worldDir = WorldRayDirection();
    float3 skyLuminance = params.sky.atmosphereParams.solar_illuminance
        * params.sky.sampleScatteredTransfer(worldDir);

    // Cosine-weighted importance sampling of Lambertian BRDF:
    // L_indirect = albedo * (1/PI) * L_i * cos(theta) / pdf = albedo * L_i
    // The 1/π term (from the BRDF) and the cos(θ) term (from the integrand) gets cancelled
    // out by the cosine - weighted distribution with pdf(ω) = cos(θ) / π
    float2 uv = (float2(pixel) + 0.5) / float2(DispatchRaysDimensions().xy);
    params.outputTexture[pixel] = float4(current.rgb + skyLuminance, current.a); // albedo is demodulated
}
