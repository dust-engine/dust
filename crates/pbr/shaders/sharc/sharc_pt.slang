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
import sharc;

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

// SharcShaderConstants now lives in pbr_common.slang as a public struct so
// that PbrPipelineParams can carry it as a field (its layout still mirrors
// crates/pbr/src/sharc.rs::SharcConstantsRaw). The standalone `[vk::binding(17,
// 0)] g_sharcConstants` below is preserved for sharc_pt's own use; it reuses
// the same descriptor slot.

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

#if SHARC_UPDATE
// Pool-driven, depth-1 Update raygen. Each invocation reads one slot of the
// candidate pool; empty slots (key == 0) no-op. Multi-bounce light transport
// is built across frames, not within a single Update call — the cache itself
// closes the loop via its terminator value, so each frame we only need:
//
//   1 ray  NEE shadow at candidate vertex      (direct light deposit)
//   1 ray  one cosine-hemisphere bounce        (depth-1 indirect)
//
// On bounce hit we read the bounce target's cache:
//   * populated cell  → SharcUpdateMiss(cached) — back-propagates the cached
//                       outgoing radiance through the throughput chain into
//                       the candidate's accumulator. The cache value already
//                       integrates all bounces beyond, so we're done.
//   * empty cell      → pushCandidate(...) — schedules an Update path for
//                       this cell next frame. No terminator: the candidate
//                       gets only direct light this frame; indirect arrives
//                       once the cascade-pushed cell converges.
//
// The cache front therefore advances by one bounce per frame from each
// "anchor" (lit, queried surface). After K frames of stable camera the
// cache is correct out to depth K.
//
// Per pool slot ray count: 2. Total Update rays per frame ≤ 2 ×
// `pool_capacity`, exactly the budget knob the host wants.
[shader("raygeneration")]
void rayGenShader() {
    uint slot = DispatchRaysIndex().x;
    uint cap = g_sharcConstants.poolCapacity;
    if (slot >= cap) return;
    if (params.sharc.keyAt(slot) == 0u) return;

    Candidate cand = params.sharc.candidateAt(slot);
    float3 candPos = cand.worldPos;
    float3 candNormal = normalize(cand.normal);
    float3 candAlbedo = cand.albedo;

    uint seed = xxhash32(uint3(slot, pc.frameIndex, g_sharcConstants.frameIndex));

    SharcParameters sharcParameters = buildSharcParameters();
    SharcState sharcState;
    SharcInit(sharcState);

    // ── Vertex 0: candidate ──────────────────────────────────────────────
    // Sun NEE → direct light deposit at the candidate.
    float3 directIrradiance = sunNEE(candPos, candNormal);
    float3 directLighting = candAlbedo * directIrradiance / kPi;
    SharcHitData candHd;
    candHd.positionWorld = candPos;
    candHd.normalWorld = candNormal;
    if (!SharcUpdateHit(sharcParameters, sharcState, candHd, directLighting,
                        rand(seed))) {
        return;  // accumulator full
    }

    // ── Bounce: one cosine-hemisphere segment ────────────────────────────
    float u0 = rand(seed);
    float u1 = rand(seed);
    float3 nextDir = cosineHemisphereSample(candNormal, u0, u1);
    SharcSetThroughput(sharcState, candAlbedo);  // Lambertian throughput

    RayDesc bounceRay;
    bounceRay.Origin = candPos + candNormal * 0.001;
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
        // Sky terminator: back-propagate sky luminance to the candidate.
        float3 skyLight = sampleSky(nextDir);
        SharcUpdateMiss(sharcParameters, sharcState, skyLight);
        return;
    }

    // Bounce hit. Try the cache as terminator first.
    float3 hitPos = bounceRay.Origin + nextDir * bp.t;
    float3 hitNormal = bp.normal;
    float3 hitAlbedo = bp.albedo;

    SharcHitData targetHd;
    targetHd.positionWorld = hitPos;
    targetHd.normalWorld = hitNormal;
    float3 cached;
    if (SharcGetCachedRadiance(sharcParameters, targetHd, cached, false)) {
        // Cache hit: the cached value is the bounce target's outgoing
        // radiance integrated over all further bounces. SharcUpdateMiss
        // back-propagates this through the throughput chain — candidate's
        // accumulator gets `directLighting + candAlbedo * cached`.
        SharcUpdateMiss(sharcParameters, sharcState, cached);
        return;
    }

    // Cache miss at bounce target. No terminator → candidate gets only
    // direct light this frame. Schedule the bounce target as a cascade
    // candidate so next frame's Update populates it; once it has samples,
    // future Updates at the candidate will close the loop via the cache
    // terminator branch above.
    float w = dot(hitAlbedo, float3(0.2126, 0.7152, 0.0722));
    uint cascadeSeed = seed ^ 0xa5cc1c4du;
    params.sharc.pushCandidate(hitPos, hitNormal, hitAlbedo, w, cascadeSeed);
}
#endif

#if SHARC_QUERY
// Legacy Query raygen — preserved for the debug-mode overlays (1..4). The
// host's `sharc_query_pass` system early-returns so this pipeline never
// actually dispatches in production; the path is kept compileable so we can
// re-enable the overlays via egui without recompiling shaders.
[shader("raygeneration")]
void rayGenShader() {
    uint2 dispatchIndex = DispatchRaysIndex().xy;
    uint2 dispatchDim = DispatchRaysDimensions().xy;
    uint2 pixel = dispatchIndex;
    uint2 screenSize = dispatchDim;

    if (params.gbuffer.albedoSRGB[pixel].a == 0.0) return;

    float3 cameraOrigin = params.uniforms.cameraOrigin();
    float3 primaryWorldPos = params.pixelPositionWorldSpace(pixel, screenSize);
    float3 primaryNormal = params.gbuffer.normalTexture[pixel].xyz;
    float3 primaryAlbedo = params.gbuffer.albedoSRGB[pixel].rgb;
    float3 incomingDir = normalize(primaryWorldPos - cameraOrigin);

    switch (g_sharcConstants.debugMode) {
        case 1:
            params.outputTexture[pixel] = float4(0.4, 0.0, 0.0, 1.0);
            return;
        case 2: {
            SharcParameters sp = buildSharcParameters();
            float3 occ = HashGridDebugOccupancy(pixel, screenSize, sp.hashGridData);
            params.outputTexture[pixel] = float4(occ, 1.0);
            return;
        }
        case 3: {
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
}
#endif

[shader("miss")]
void primaryMissShader(inout SharcHitPayload payload) {
    payload.t = -1.0;
}

[shader("miss")]
void shadowMissShader(inout SharcShadowPayload payload) {
    payload.occluded = 0;
}
