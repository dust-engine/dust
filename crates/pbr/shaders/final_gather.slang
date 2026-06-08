import pbr_common;
import random;

#define HASH_GRID_ENABLE_64_BIT_ATOMICS 1
#define HASH_GRID_COMPACT 0
#define SHARC_ENABLE_GLSL 0
#define SHARC_ENABLE_RESPONSIVE_LIGHTING 0
#define SHARC_ENABLE_SH_ENCODING 0
#define RW_STRUCTURED_BUFFER(name, type) RWStructuredBuffer<type> name
#define BUFFER_AT_OFFSET(name, offset) name[offset]
#include "SharcCommon.h"

struct [raypayload] GatherPayload {
    float dummy : read(caller) : write(caller, miss);
}

// Push constant block shared by the final-gather ray-gen. `frameIndex` drives
// the per-frame sampling seed; the `sharcDebug*` fields drive the Phase 1
// hash-grid debug overlay (see PbrPipelineParams / pbr.playout.ron). When
// `sharcDebugEnabled` is non-zero the ray-gen paints the SHARC colored-hash
// instead of tracing an indirect bounce.
struct FinalGatherPush {
    uint frameIndex;
    uint sharcDebugEnabled;
    float sharcSceneScale;
    float sharcBrightness;
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

    // Skip pixels with no opaque surface to shade. The primary RT pass writes
    // albedo.a = 1 only for real hits; sky and UI both write 0.
    if (params.gbuffer.albedoSRGB[pixel].a == 0.0) {
        return;
    }

    // Reconstruct world position
    float3 worldPos = params.pixelPositionWorldSpace(pixel, DispatchRaysDimensions().xy);
    float3 N = params.gbuffer.normalTexture[pixel].xyz;

    // Phase 1 SHARC debug overlay: replace the indirect bounce with the spatial
    // hash color so we can verify grid layout / scale against surfaces.
    if (pc.sharcDebugEnabled != 0) {
        HashGridParameters gridParameters;
        gridParameters.cameraPosition = params.uniforms.cameraOrigin();
        gridParameters.logarithmBase = SHARC_GRID_LOGARITHM_BASE;
        gridParameters.sceneScale = pc.sharcSceneScale;
        gridParameters.levelBias = SHARC_GRID_LEVEL_BIAS;

        float3 color = HashGridDebugColoredHash(worldPos, N, gridParameters);
        params.outputTexture[pixel] = float4(color * pc.sharcBrightness, 1.0);
        return;
    }

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
