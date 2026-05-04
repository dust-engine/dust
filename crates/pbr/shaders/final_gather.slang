import pbr_common;
import random;

struct [raypayload] GatherPayload {
    float dummy : read(caller) : write(caller, miss);
}

[vk::push_constant] uniform uint frameIndex;

// xxhash32 over a 4D uint key. From Jarzynski & Olano,
// "Hash Functions for GPU Rendering" (JCGT 9.3, 2020). Passes BigCrush with
// few failures, which is what DLSS-RR's sampling guidance asks for. Use a
// distinct sample-dimension key in `.w` per output dimension to keep the
// low-dimensional projections decorrelated.
static const uint XXH_PRIME32_2 = 2246822519u;
static const uint XXH_PRIME32_3 = 3266489917u;
static const uint XXH_PRIME32_4 = 668265263u;
static const uint XXH_PRIME32_5 = 374761393u;

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

    uint seed = xxhash32(uint3(pixel.x, pixel.y, frameIndex));
    float u0 = rand(seed);
    float u1 = rand(seed);
    float r = sqrt(u0);
    float phi = 6.2831853071795864 * u1;
    float3 tangentDir = float3(r * cos(phi), r * sin(phi), sqrt(1.0 - u0));

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
    //   L_indirect = albedo * (1/π) * L_i * cos(θ) / pdf = albedo * L_i
    // The 1/π (BRDF) and cos(θ) (integrand) cancel against pdf(ω) = cos(θ)/π.
    float3 albedo = params.gbuffer.albedoSRGB[pixel].rgb;
    params.outputTexture[pixel] = float4(current.rgb + albedo * skyLuminance, current.a);
}
