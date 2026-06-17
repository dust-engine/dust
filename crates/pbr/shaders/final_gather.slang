import pbr_common;
import random;

struct [raypayload] GatherPayload {
    float dummy : read(caller) : write(caller, miss);
}

// Push constant for the final-gather ray-gen: just the per-frame sampling seed.
// (The SHARC hash-grid debug overlay used to ride along here; it now lives in
// the primary pass — see vox_pbr.slang / SharcShaderConstants::debugMode.)
struct FinalGatherPush {
    uint frameIndex;
}
[vk::push_constant] uniform FinalGatherPush pc;

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
    uint missShaderIndex = params.gbuffer.albedoUint[pixel].a;

    // albedoUint.a is a per-pixel miss shader index written by the primary pass:
    // real hits store 0; sky/UI store 0xFF as a "no surface" sentinel. Skip the
    // sentinel pixels; the rest is passed straight to TraceRay's MissShaderIndex.
    if (missShaderIndex == 0xFF) {
        return;
    }

    // Reconstruct world position
    float3 worldPos = params.pixelPositionWorldSpace(pixel, DispatchRaysDimensions().xy);
    float3 N = params.gbuffer.normalTexture[pixel].xyz;

    uint seed = xxhash32(uint3(pixel.x, pixel.y, pc.frameIndex));
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
        RAY_FLAG_FORCE_OPAQUE,
        ~0,
        0, 1, missShaderIndex,
        ray, payload
    );

    // Demodulate the specular-albedo G-buffer for DLSS-RR. The shading passes
    // (shadow + this gather's CHS/miss) consume the stored F0; this ray-gen is
    // the last G-buffer reader before DLSS, so once the trace above returns we
    // overwrite F0 in place with the pre-integrated environment BRDF
    // EnvBRDFApprox2(F0, roughness², NoV) — the specular "albedo" DLSS-RR expects
    // for demodulation. Per-pixel, same-thread read-before-write: no hazard.
    float3 F0 = params.gbuffer.specularAlbedo[pixel].rgb;
    float roughness = params.gbuffer.normalTexture[pixel].w;
    float3 V = -params.uniforms.primaryRayDirWorldSpace(pixel, DispatchRaysDimensions().xy);
    params.gbuffer.specularAlbedo[pixel] =
        float4(EnvBRDFApprox2(F0, roughness * roughness, dot(N, V)), 1.0);
}

[shader("miss")]
void finalGatherMissShader() {
    uint2 pixel = DispatchRaysIndex().xy;
    float4 current = params.outputTexture[pixel];

    float3 worldDir = WorldRayDirection();
    float3 skyLuminance = params.sky.atmosphereParams.solar_illuminance
        * params.sky.sampleScatteredTransfer(worldDir);

    // Single-sample BRDF estimator over the cosine-sampled gather ray:
    //   weight = f·cos/pdf = diffuseAlbedo + π·f_spec   (pdf = cos/π cancels cos)
    // The diffuse term is the original Lambertian cancellation; the specular term
    // evaluates GGX for the primary surface (N, V) toward this sky sample wi,
    // reusing the same ray — no extra trace. See pbr_common::cosineSpecularWeight.
    float3 albedo = params.gbuffer.albedoSRGB[pixel].rgb;
    float3 N = params.gbuffer.normalTexture[pixel].xyz;
    float roughness = params.gbuffer.normalTexture[pixel].w;
    float3 F0 = params.gbuffer.specularAlbedo[pixel].rgb;
    float3 V = -params.uniforms.primaryRayDirWorldSpace(pixel, DispatchRaysDimensions().xy);
    float3 weight = albedo + cosineSpecularWeight(N, V, worldDir, roughness, F0);
    params.outputTexture[pixel] = float4(current.rgb + weight * skyLuminance, current.a);
}
