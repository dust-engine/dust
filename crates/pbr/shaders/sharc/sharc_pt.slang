// SHARC path tracer — sparse cache-population pass ("Update" in SDK terms).
//
// Writes ONLY into the SHARC buffers (AGENTS.md §15); the visible indirect
// bounce is still served by final_gather. Update is what keeps the cache fresh
// across frames so that downstream gather can sample from it.

import pbr_common;
import random;
import sharc;

// The cache is addressed through `params.sharc` key-based helpers
// (cellKey / updateHitKey / getCachedRadianceByKey / addRadianceToSlot) — see
// sharc.slang. This pass no longer includes SharcCommon.h or builds its own
// SharcParameters: candidates carry a precomputed voxel-identity key, so the
// SDK's position→spatial-hash machinery is bypassed entirely.

struct PushConstants {
    uint frameIndex;
}
[vk::push_constant] uniform PushConstants pc;

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

// Pool-driven, depth-1 Update raygen. Each invocation reads one slot of the
// candidate pool; empty slots (key == 0) no-op. Multi-bounce light transport
// is built across frames, not within a single Update call — the cache itself
// closes the loop via its terminator value, so each frame we only need:
//
//   1 ray  NEE shadow at candidate vertex      (direct light deposit)
//   1 ray  one cosine-hemisphere bounce        (depth-1 indirect)
//
// On bounce hit we read the bounce target's cache (by its key from the
// payload):
//   * populated cell  → addRadianceToSlot(cached) — back-propagates the cached
//                       outgoing radiance through the throughput into the
//                       candidate's slot. The cache value already integrates
//                       all bounces beyond, so we're done.
//   * empty cell      → pushCandidate(...) — schedules an Update path for this
//                       cell next frame. No terminator: the candidate gets only
//                       direct light this frame; indirect arrives once the
//                       cascade-pushed cell converges.
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
    uint cap = params.sharc.g_sharcConstants.poolCapacity;
    if (slot >= cap) return;
    if (params.sharc.keyAt(slot) == 0u) return;

    Candidate cand = params.sharc.candidateAt(slot);
    float3 candPos = cand.worldPos;
    float3 candNormal = normalize(cand.normal);
    float3 candAlbedo = cand.albedo;
    uint64_t candKey = cand.key;

    uint seed = xxhash32(uint3(slot, pc.frameIndex, params.sharc.g_sharcConstants.frameIndex));

    // ── Vertex 0: candidate ─ Sun NEE → direct light deposit into its cell. ─
    float3 directIrradiance = sunNEE(candPos, candNormal);
    float3 directLighting = candAlbedo * directIrradiance / kPi;
    uint candSlot;
    if (!params.sharc.updateHitKey(candKey, directLighting, candSlot)) {
        return;  // probe bucket full
    }

    // ── Bounce: one cosine-hemisphere segment ────────────────────────────
    // Lambertian throughput candidate→bounce is `candAlbedo` (the cos/pdf
    // cancels under cosine-importance sampling).
    float u0 = rand(seed);
    float u1 = rand(seed);
    float3 nextDir = cosineHemisphereSample(candNormal, u0, u1);

    RayDesc bounceRay;
    bounceRay.Origin = candPos + candNormal * 0.001;
    bounceRay.Direction = nextDir;
    bounceRay.TMin = 0.0;
    bounceRay.TMax = params.uniforms.far;
    SharcHitPayload bp;
    bp.t = -1.0;
    bp.albedo = float3(0, 0, 0);
    bp.normal = float3(0, 0, 0);
    bp.key = 0;
    TraceRay(
        params.sceneBVH,
        RAY_FLAG_FORCE_OPAQUE,
        ~0, 0, 1, 0,
        bounceRay, bp
    );

    if (bp.t < 0.0) {
        // Sky terminator: back-propagate sky luminance through the throughput.
        params.sharc.addRadianceToSlot(candSlot, sampleSky(nextDir), candAlbedo);
        return;
    }

    // Bounce hit. Try the bounce target's cache (by key) as terminator.
    float3 cached;
    float bounceSampleNum;
    if (params.sharc.getCachedRadianceByKey(bp.key, cached, bounceSampleNum)) {
        // Cached value already integrates all further bounces — fold it into
        // the candidate's slot scaled by throughput.
        params.sharc.addRadianceToSlot(candSlot, cached, candAlbedo);
        return;
    }

    // Cache miss at bounce target. No terminator → candidate gets only direct
    // light this frame. Schedule the bounce target as a cascade candidate so
    // next frame's Update populates it; once it has samples, future Updates at
    // the candidate close the loop via the terminator branch above.
    float3 hitPos = bounceRay.Origin + nextDir * bp.t;
    float w = dot(bp.albedo, float3(0.2126, 0.7152, 0.0722));
    uint cascadeSeed = seed ^ 0xa5cc1c4du;
    params.sharc.pushCandidate(hitPos, bp.normal, bp.albedo, bp.key, w, cascadeSeed);
}

[shader("miss")]
void primaryMissShader(inout SharcHitPayload payload) {
    payload.t = -1.0;
}

[shader("miss")]
void shadowMissShader(inout SharcShadowPayload payload) {
    payload.occluded = 0;
}
