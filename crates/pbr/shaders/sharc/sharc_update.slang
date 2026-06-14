// SHARC path tracer — sparse cache-population pass ("Update" in SDK terms).
//
// Writes ONLY into the SHARC buffers; the visible indirect bounce is still
// served by final_gather. Update keeps the cache fresh across frames so the
// downstream gather can sample from it.
//
// Minimal-payload design (Intel RTRT best practice — keep payloads as small as
// possible, recompute what you can): the only thing on the ray is the source
// candidate's resolved cache slot (4 B). The ray-gen opens the cell once (a
// hash-grid find-or-insert) and counts the candidate's single per-frame sample,
// then hands the slot down so each deposit stage folds its contribution in with
// a direct indexed atomic add instead of repeating the bucket walk:
//
//   ray-gen            open cell (find-or-insert + count 1); slot → payload;
//                      trace 1 shadow + 1 bounce ray
//   shadow miss        sun visible → deposit direct lighting    (into slot)
//   bounce miss        ray escaped → deposit sky                (into slot)
//   bounce closesthit  (vox_sharc_update) deposit cached terminator + re-seed
//                      cascade                                  (into slot)
//
// The sample count is owned by the ray-gen — unconditional, so every candidate
// counts exactly one sample regardless of sun visibility or bounce outcome,
// including fully shadowed cells whose shadow-miss deposit never runs. All
// lighting deposits are additive (count 0). Each stage reloads albedo/normal/pos
// from the candidate pool by DispatchRaysIndex().x — cheap, so it stays off the
// payload; only the slot, costly to recompute, rides it. See openCellForUpdate /
// addRadianceToSlot in sharc.slang.

import pbr_common;
import random;
import sharc;

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

float3 sampleSky(float3 dir) {
    return params.sky.atmosphereParams.solar_illuminance
        * params.sky.sampleScatteredTransfer(dir);
}

[shader("raygeneration")]
void rayGenShader() {
    uint slot = DispatchRaysIndex().x;
    uint cap = params.sharc.g_sharcConstants.poolCapacity;
    if (slot >= cap) return;
    if (params.sharc.keyAt(slot) == 0u) return;

    Candidate cand = params.sharc.candidateAt(slot);
    float3 candPos = cand.worldPos;
    float3 candNormal = normalize(cand.normal);

    // Open the cell once. If the probe bucket is full there's nowhere to
    // deposit — drop the candidate.
    SharcRayPayload pl;
    if (!params.sharc.openCellForUpdate(cand.key, pl.cacheSlot)) return;

    // ── Sun NEE shadow ray ───────────────────────────────────────────────
    // shadowMissShader deposits the direct lighting when the sun is visible
    // (miss). On occlusion the ACCEPT_FIRST_HIT | SKIP_CLOSEST_HIT flags run no
    // shader at all → nothing is deposited. Skip the trace on back-facing normals.
    float3 sunDir = params.sky.atmosphereParams.sun_direction;
    if (dot(candNormal, sunDir) > 0.0) {
        RayDesc shadow;
        shadow.Origin = candPos + candNormal * 0.01;
        shadow.Direction = sunDir;
        shadow.TMin = 0.0;
        shadow.TMax = params.uniforms.far;
        TraceRay(
            params.sceneBVH,
            RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH
                | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_FORCE_OPAQUE,
            ~0, 0, 1, 1, shadow, pl
        );
    }

    // ── One cosine-hemisphere bounce ─────────────────────────────────────
    // bounceMissShader (sky) or vox_sharc_update's closest-hit (cached terminator)
    // deposits the indirect contribution into pl.cacheSlot. The Lambertian
    // throughput (cand.albedo) is applied at the deposit site.
    uint seed = xxhash32(uint3(slot, pc.frameIndex,
                               params.sharc.g_sharcConstants.frameIndex));
    float u0 = rand(seed);
    float u1 = rand(seed);
    float3 nextDir = cosineHemisphereSample(candNormal, u0, u1);

    RayDesc bounceRay;
    bounceRay.Origin = candPos + candNormal * 0.001;
    bounceRay.Direction = nextDir;
    bounceRay.TMin = 0.0;
    bounceRay.TMax = params.uniforms.far;
    TraceRay(params.sceneBVH, RAY_FLAG_FORCE_OPAQUE, ~0, 0, 1, 0, bounceRay, pl);
}

// Bounce miss — the bounce ray escaped to sky. Deposit sky luminance into the
// candidate's cell (slot from the payload), scaled by its Lambertian throughput.
[shader("miss")]
void bounceMissShader(inout SharcRayPayload payload) {
    uint slot = DispatchRaysIndex().x;
    Candidate cand = params.sharc.candidateAt(slot);
    float3 sky = sampleSky(WorldRayDirection());
    params.sharc.addRadianceToSlot(payload.cacheSlot, sky, cand.albedo);
}

// Shadow miss — the sun is visible from the candidate. Deposit direct Lambertian
// outgoing radiance into the candidate's cell (slot from the payload). The
// sample was already counted in the ray-gen, so this is purely additive.
[shader("miss")]
void shadowMissShader(inout SharcRayPayload payload) {
    uint slot = DispatchRaysIndex().x;
    Candidate cand = params.sharc.candidateAt(slot);
    float3 N = normalize(cand.normal);
    float3 sunDir = params.sky.atmosphereParams.sun_direction;
    float ndotL = max(0.0, dot(N, sunDir));
    if (ndotL <= 0.0) return;
    float3 transmittance = params.sky.sampleTransmittanceAt(cand.worldPos);
    float3 irradiance =
        params.sky.atmosphereParams.solar_illuminance * transmittance * ndotL;
    float3 directLighting = cand.albedo * irradiance / kPi;
    params.sharc.addRadianceToSlot(payload.cacheSlot, directLighting,
                                   float3(1.0, 1.0, 1.0));
}
