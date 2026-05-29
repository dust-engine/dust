// SHARC path tracer (Update + Query permutations).
//
// Compiled with one of:
//   SHARC_UPDATE = 1, SHARC_QUERY = 0  ⇒  sparse cache-population pass
//   SHARC_UPDATE = 0, SHARC_QUERY = 1  ⇒  full-res render pass with cache early-out
// Both shaders share this single source so the bounce loop, NEE, and hit
// classification stay identical across permutations (AGENTS.md §11, §14).
//
// The renderer keeps its existing primary-RT + shadow passes; the Query
// permutation REPLACES the current sky-only final_gather as the indirect
// integrator. Update writes ONLY into the SHARC buffers (AGENTS.md §15), Query
// adds its gathered indirect on top of the HDR that the shadow pass produced.

import pbr_common;
import random;

#ifndef SHARC_UPDATE
#define SHARC_UPDATE 0
#endif
#ifndef SHARC_QUERY
#define SHARC_QUERY 0
#endif

// SDK plumbing. AGENTS.md §14.12: for HLSL/Slang, include SharcCommon.h
// directly — do NOT pull in SharcGlslHelpers.h.
#define HASH_GRID_ENABLE_64_BIT_ATOMICS 1
#define HASH_GRID_COMPACT 0
#define SHARC_ENABLE_GLSL 0
#define SHARC_ENABLE_RESPONSIVE_LIGHTING 0
#define SHARC_ENABLE_SH_ENCODING 0
#define RW_STRUCTURED_BUFFER(name, type) RWStructuredBuffer<type> name
#define BUFFER_AT_OFFSET(name, offset) name[offset]
#include "SharcCommon.h"

// Per-frame SHARC constants (matches the Rust SharcConstants in sharc.rs).
struct SharcShaderConstants {
    float3 cameraPosition;
    float sceneScale;
    float3 cameraPositionPrev;
    float radianceScale;
    float roughnessMin;
    uint accumulationFrameNum;
    uint staleFrameNum;
    uint entriesNum;
    uint frameIndex;
    uint downscaleFactor;
    uint debugMode;
    uint _pad;
}

// SHARC bindings co-locate with `params` in descriptor set 0 (see
// pbr_common.slang for why the ParameterBlock is pinned to set 0). Bindings
// 14..17 fall after the 14 fields auto-assigned to PbrPipelineParams
// (0..13) — single push-descriptor set, stable across Update and Query
// (AGENTS.md §9).
[vk::binding(14, 0)] RWStructuredBuffer<HashGridKey> g_sharcHashEntries;
[vk::binding(15, 0)] RWStructuredBuffer<SharcAccumulationData> g_sharcAccumulation;
[vk::binding(16, 0)] RWStructuredBuffer<SharcPackedData> g_sharcResolved;
[vk::binding(17, 0)] ConstantBuffer<SharcShaderConstants> g_sharcConstants;

SharcParameters buildSharcParameters() {
    SharcParameters p;
    p.hashGridParameters.cameraPosition = g_sharcConstants.cameraPosition;
    p.hashGridParameters.sceneScale = g_sharcConstants.sceneScale;
    p.hashGridParameters.logarithmBase = SHARC_GRID_LOGARITHM_BASE;
    p.hashGridParameters.levelBias = SHARC_GRID_LEVEL_BIAS;
    p.hashGridData.capacity = g_sharcConstants.entriesNum;
    p.hashGridData.hashEntriesBuffer = g_sharcHashEntries;
    p.accumulationBuffer = g_sharcAccumulation;
    p.resolvedBuffer = g_sharcResolved;
    p.radianceScale = g_sharcConstants.radianceScale;
    return p;
}

struct PushConstants {
    uint frameIndex;
}
[vk::push_constant] uniform PushConstants pc;

static const uint kMaxBounces = 4;
static const float kPi = 3.14159265358979;

void buildTBN(float3 N, out float3 T, out float3 B) {
    if (abs(N.y) < 0.999) T = normalize(cross(N, float3(0, 1, 0)));
    else                   T = normalize(cross(N, float3(1, 0, 0)));
    B = cross(N, T);
}

float3 cosineHemisphereSample(float3 N, float u0, float u1) {
    float r = sqrt(u0);
    float phi = 2.0 * kPi * u1;
    float3 local = float3(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u0)));
    float3 T, B; buildTBN(N, T, B);
    return normalize(T * local.x + B * local.y + N * local.z);
}

// Sun-direct lighting at the supplied shading point. Returns
// `sun_illuminance * transmittance * cos(N·L) * visibility` (irradiance), so
// the caller multiplies by `albedo / pi` to get the Lambertian outgoing
// radiance. Uses miss index 1 (the shadow miss) and skips the closest hit.
float3 sunNEE(float3 worldPos, float3 N) {
    float3 sunDir = params.sky.atmosphereParams.sun_direction;
    float ndotL = max(0.0, dot(N, sunDir));
    if (ndotL <= 0.0) return float3(0, 0, 0);

    RayDesc shadow;
    shadow.Origin = worldPos + N * 0.01;
    shadow.Direction = sunDir;
    shadow.TMin = 0.0;
    shadow.TMax = params.uniforms.far;

    SharcShadowPayload sp;
    sp.occluded = 1;
    TraceRay(
        params.sceneBVH,
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_FORCE_OPAQUE,
        ~0, 0, 1, 1, shadow, sp
    );
    if (sp.occluded != 0) return float3(0, 0, 0);

    float3 transmittance = params.sky.sampleTransmittanceAt(worldPos);
    return params.sky.atmosphereParams.solar_illuminance * transmittance * ndotL;
}

float3 sampleSky(float3 dir) {
    return params.sky.atmosphereParams.solar_illuminance
        * params.sky.sampleScatteredTransfer(dir);
}

[shader("raygeneration")]
void rayGenShader() {
    uint2 dispatchIndex = DispatchRaysIndex().xy;
    uint2 dispatchDim = DispatchRaysDimensions().xy;

    uint2 pixel;
    uint2 screenSize;
#if SHARC_UPDATE
    // Sparse coverage: launch is downscaled; map (tileX, tileY) → one of
    // downscale² pixels per tile, frame-rotated so the cache fills the whole
    // frustum over `downscale²` frames (AGENTS.md §14.3 "Acceptable strategy
    // 1"). NEVER write to the top-left subrect.
    uint downscale = max(g_sharcConstants.downscaleFactor, 1u);
    screenSize = dispatchDim * downscale;
    uint subSeed = xxhash32(uint3(dispatchIndex.x, dispatchIndex.y,
                                  g_sharcConstants.frameIndex));
    uint sub = subSeed % (downscale * downscale);
    pixel = dispatchIndex * downscale + uint2(sub % downscale, sub / downscale);
    if (pixel.x >= screenSize.x || pixel.y >= screenSize.y) return;
#else
    pixel = dispatchIndex;
    screenSize = dispatchDim;
#endif

    // Skip sky/UI pixels (no primary surface).
    if (params.gbuffer.albedoSRGB[pixel].a == 0.0) return;

    // Primary vertex comes from the G-buffer — we never re-trace the camera
    // ray for SHARC (split-primary architecture per AGENTS.md §14.2 case 2).
    float3 cameraOrigin = params.uniforms.cameraOrigin();
    float3 primaryWorldPos = params.pixelPositionWorldSpace(pixel, screenSize);
    float3 primaryNormal = params.gbuffer.normalTexture[pixel].xyz;
    float3 primaryAlbedo = params.gbuffer.albedoSRGB[pixel].rgb;
    float3 incomingDir = normalize(primaryWorldPos - cameraOrigin);

#if SHARC_QUERY
    // SHARC debug visualization modes (set via SharcConfig::debug_mode on the
    // host). All modes replace the HDR output rather than adding so the
    // overlay is unambiguous. NOTE: only emitted in the Query permutation —
    // Update never writes the final color (AGENTS.md §15).
    switch (g_sharcConstants.debugMode) {
        case 1: {
            // Sanity check that Query's ray-gen is running at all. If you see
            // a flat red wash on shadowed pixels, the pass dispatches; if
            // shadows stay black with this mode on, the system isn't running.
            params.outputTexture[pixel] = float4(0.4, 0.0, 0.0, 1.0);
            return;
        }
        case 2: {
            // HashGridDebugOccupancy — paints the entries buffer as a tiled
            // heatmap (green = populated, gold = responsive). All black =
            // empty cache (Update isn't writing).
            SharcParameters sp = buildSharcParameters();
            float3 occ = HashGridDebugOccupancy(pixel, screenSize, sp.hashGridData);
            params.outputTexture[pixel] = float4(occ, 1.0);
            return;
        }
        case 3: {
            // HashGridDebugColoredHash — colored-hash at the primary hit
            // (same view as Phase 1's final_gather overlay).
            HashGridParameters gp;
            gp.cameraPosition = cameraOrigin;
            gp.sceneScale = g_sharcConstants.sceneScale;
            gp.logarithmBase = SHARC_GRID_LOGARITHM_BASE;
            gp.levelBias = SHARC_GRID_LEVEL_BIAS;
            float3 col = HashGridDebugColoredHash(primaryWorldPos, primaryNormal, gp);
            params.outputTexture[pixel] = float4(col, 1.0);
            return;
        }
        case 4: {
            // Cache-content overlay: look up the resolved radiance at the
            // primary worldPos and paint it. Magenta = no entry, white-tinted
            // = entry exists but accumulatedSampleNum == 0, otherwise the
            // cached luminance. This tells us whether Update / Resolve are
            // producing usable data.
            SharcParameters sp = buildSharcParameters();
            SharcHitData hd;
            hd.positionWorld = primaryWorldPos;
            hd.normalWorld = primaryNormal;
            float3 cached;
            if (SharcGetCachedRadiance(sp, hd, cached, false)) {
                params.outputTexture[pixel] = float4(cached, 1.0);
            } else {
                params.outputTexture[pixel] = float4(1.0, 0.0, 1.0, 1.0);
            }
            return;
        }
    }
#endif

    uint seed = xxhash32(uint3(pixel.x, pixel.y, pc.frameIndex));

#if SHARC_UPDATE
    SharcParameters sharcParameters = buildSharcParameters();
    SharcState sharcState;
    SharcInit(sharcState);
#endif

    // Update accumulates each segment independently (SDK guidance);
    // Query/Reference propagate throughput across the path.
    float3 sampleRadiance = float3(0, 0, 0);
    float3 throughput = float3(1, 1, 1);

    float3 hitPos = primaryWorldPos;
    float3 hitNormal = primaryNormal;
    float3 hitAlbedo = primaryAlbedo;
    bool isPrimary = true;
    float lastSegmentLength = 0.0;

    for (uint bounce = 0; bounce < kMaxBounces; ++bounce) {
        // Sun NEE — works at primary and indirect bounces. The bounce-hit
        // direct lighting is the new contribution that final_gather lacks;
        // primary-hit direct is already in HDR via the shadow pass, so for
        // Query/Reference we only fold it into `sampleRadiance` on indirect
        // bounces. Update still submits direct at every vertex so the cache
        // is self-contained.
        float3 directIrradiance = sunNEE(hitPos, hitNormal);
        float3 directLighting = hitAlbedo * directIrradiance / kPi;

#if SHARC_QUERY
        // Indirect-bounce cache query (AGENTS.md §14.8). Skip primary vertex
        // — those samples are owned by the shadow pass. The roughness/
        // footprint gate is vacuous for our diffuse Lambertian voxels (cone
        // spread spans the hemisphere) so it's not applied. The segment-
        // length gate is intentionally OFF here: with the cache cold or
        // sparsely populated, gating short segments drops the *only* indirect
        // contribution shadowed pixels can get, painting them black.
        if (!isPrimary) {
            SharcParameters sp = buildSharcParameters();
            SharcHitData hd;
            hd.positionWorld = hitPos;
            hd.normalWorld = hitNormal;
            float3 cached;
            if (SharcGetCachedRadiance(sp, hd, cached, false)) {
                sampleRadiance += throughput * cached;
                break;
            }
        }
#endif

#if SHARC_UPDATE
        SharcHitData hd;
        hd.positionWorld = hitPos;
        hd.normalWorld = hitNormal;
        if (!SharcUpdateHit(sharcParameters, sharcState, hd, directLighting, rand(seed))) {
            break;
        }
#else
        // Direct lighting only feeds the additive HDR output on indirect
        // bounces (see comment above).
        if (!isPrimary) {
            sampleRadiance += throughput * directLighting;
        }
#endif

        // Sample next direction (Lambertian cosine-hemisphere).
        float u0 = rand(seed);
        float u1 = rand(seed);
        float3 nextDir = cosineHemisphereSample(hitNormal, u0, u1);
        // Lambertian throughput cancellation: (albedo / pi) * cos / (cos / pi) = albedo.
        float3 segmentThroughput = hitAlbedo;

#if SHARC_UPDATE
        // SHARC accumulates each segment independently. Submit segment
        // throughput, then reset throughput so the next segment starts at
        // unit weight (SDK comment on SharcSetThroughput).
        SharcSetThroughput(sharcState, segmentThroughput);
        throughput = float3(1, 1, 1);
        sampleRadiance = float3(0, 0, 0);
#else
        throughput *= segmentThroughput;
#endif

        RayDesc bounceRay;
        bounceRay.Origin = hitPos + hitNormal * 0.001;
        bounceRay.Direction = nextDir;
        bounceRay.TMin = 0.0;
        bounceRay.TMax = params.uniforms.far;
        SharcHitPayload bp;
        bp.t = -1.0;
        bp.albedo = float3(0, 0, 0);
        bp.normal = float3(0, 0, 0);
        TraceRay(
            params.sceneBVH,
            RAY_FLAG_FORCE_OPAQUE,
            ~0, 0, 1, 0,
            bounceRay, bp
        );

        if (bp.t < 0.0) {
            float3 skyLight = sampleSky(nextDir);
#if SHARC_UPDATE
            SharcUpdateMiss(sharcParameters, sharcState, skyLight);
#else
            sampleRadiance += throughput * skyLight;
#endif
            break;
        }

        hitPos = bounceRay.Origin + nextDir * bp.t;
        hitNormal = bp.normal;
        hitAlbedo = bp.albedo;
        incomingDir = nextDir;
        lastSegmentLength = bp.t;
        isPrimary = false;
    }

#if !SHARC_UPDATE
    // Add gathered indirect on top of the direct/sky contribution that the
    // primary RT + shadow passes already wrote. Same contract as the existing
    // final_gather.slang miss shader.
    float4 current = params.outputTexture[pixel];
    params.outputTexture[pixel] = float4(current.rgb + sampleRadiance, current.a);
#endif
}

[shader("miss")]
void primaryMissShader(inout SharcHitPayload payload) {
    payload.t = -1.0;
}

[shader("miss")]
void shadowMissShader(inout SharcShadowPayload payload) {
    payload.occluded = 0;
}
