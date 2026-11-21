// SHARC path tracer — sparse cache-population pass ("Update").
//
// Writes ONLY into the SHARC buffers; the visible indirect bounce is still
// served by final_gather. Update keeps the cache fresh across frames so the
// downstream gather can sample from it.
//
// Minimal-payload design (Intel RTRT best practice — keep payloads as small as
// possible, recompute what you can): the ray carries the source candidate's
// resolved cache slot and, for shadow rays, whether they got through. The
// ray-gen opens the cell once (a hash-map find-or-insert) and counts the
// candidate's single per-frame sample, then deposits each contribution with a
// direct indexed atomic add instead of repeating the bucket walk:
//
//   ray-gen            open cell (find-or-insert + count 1); slot → payload;
//                      trace a shadow ray to the sun and one to a point on an
//                      emissive face, depositing the light of each that gets
//                      through; trace 1 bounce ray
//   shadow miss        nothing blocks the ray → mark it visible
//   bounce miss        ray escaped → deposit sky                (into slot)
//   bounce closesthit  (vox_sharc_update) deposit cached terminator + re-seed
//                      cascade                                  (into slot)
//
// The sample count is owned by the ray-gen — unconditional, so every candidate
// counts exactly one sample regardless of what its rays find, including fully
// shadowed cells that deposit nothing. All lighting deposits are additive
// (count 0). The miss and closest-hit stages reload albedo/normal/pos from the
// candidate pool by DispatchRaysIndex().x — cheap, so it stays off the payload;
// only the slot, costly to recompute, rides it. See openCellForUpdate /
// addRadianceToSlot in sharc.slang.
//
// Emissive voxels light the cell through its emitter reservoir
// (`EmitterReservoir` in emitters.slang, `SharcReservoir` in sharc.slang): one
// point on an emissive face, resampled from the point the cell kept and fresh
// candidates drawn by power, with a shadow ray to it. When that's on (`SharcShaderConstants::samplesEmitters`), the bounce
// ray's closest-hit leaves out the emission of what it hits, so it isn't
// counted twice; when it's off, emitters light the cell only when the bounce
// ray happens to hit one.

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

// Whether nothing blocks `ray`. The ACCEPT_FIRST_HIT | SKIP_CLOSEST_HIT flags
// run no shader on a hit, so only shadowMissShader can mark it visible.
bool traceShadowRay(RayDesc ray, inout SharcRayPayload payload) {
    payload.visible = 0;
    TraceRay(
        params.sceneBVH,
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH
            | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_FORCE_OPAQUE,
        ~0, 0, 1, 1, ray, payload
    );
    return payload.visible != 0;
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
    bool first;
    if (!params.sharc.openCellForUpdate(cand.key, pl.cacheSlot, first)) return;

    uint seed = xxhash32(uint3(slot, pc.frameIndex,
                               params.sharc.g_sharcConstants.frameIndex));

    // ── Sun ──────────────────────────────────────────────────────────────
    // Direct Lambertian outgoing radiance, when nothing blocks the sun. Skip
    // the trace on back-facing normals.
    float3 sunDir = params.sky.atmosphereParams.sun_direction;
    float sunCos = dot(candNormal, sunDir);
    if (sunCos > 0.0) {
        RayDesc shadow;
        shadow.Origin = candPos + candNormal * 0.01;
        shadow.Direction = sunDir;
        shadow.TMin = 0.0;
        shadow.TMax = params.uniforms.far;
        if (traceShadowRay(shadow, pl)) {
            float3 samplePosFromCamera = candPos - params.uniforms.cameraOrigin();
            float3 transmittance = params.sky.sampleTransmittanceAt(samplePosFromCamera);
            float3 irradiance =
                params.sky.atmosphereParams.solar_illuminance * transmittance * sunCos;
            params.sharc.addRadianceToSlot(pl.cacheSlot, cand.albedo * irradiance / kPi,
                                           float3(1.0, 1.0, 1.0));
        }
    }

    // ── Emissive voxels ──────────────────────────────────────────────────
    // ReSTIR on the cell's reservoir. Stream in the reservoir the cell kept
    // from earlier frames, re-evaluated here, and fresh candidates drawn by
    // power; keep one sample and trace a shadow ray to it. When nothing blocks
    // it, deposit the Lambertian outgoing radiance it gives,
    // (albedo / π) · L_e · cosθ · cosθ' / d² · W. The cell's first Update this
    // frame keeps the reservoir for the next frame; its other Updates read the
    // same kept reservoir and keep nothing.
    //
    // The reservoir is kept whether or not the ray got through. Visibility
    // reuse biases the cache either way: keeping a blocked sample with W = 0
    // makes it stand for up to 20 × candidates of nothing, darkening the next
    // frames by up to 21×; dropping it keeps only samples that got through,
    // whose W still counts the light of the blocked ones, brightening the
    // cache (by 24% in the emissive room).
    SharcShaderConstants constants = params.sharc.g_sharcConstants;
    if (constants.samplesEmitters) {
        uint rng = xxhash32(uint3(seed, slot, 0x9e3779b9u));
        EmitterReservoir reservoir = EmitterReservoir();
        bool reuse = (constants.flags & SHARC_FLAG_REUSE_EMITTER_SAMPLES) != 0;
        if (reuse) {
            SharcReservoir kept = params.sharc.keptReservoir(pl.cacheSlot);
            // ReSTIR caps the history at 20 times the fresh candidates.
            float maxCount = 20.0 * float(constants.emitterCandidates);
            reservoir.merge(constants.emitterInstances, constants.emitterInstanceCount,
                            kept.sample, kept.W, kept.count, maxCount, candPos, candNormal, rng);
        }
        reservoir.addCandidates(constants.emitterInstances, constants.emitterInstanceCount,
                                constants.emitterPower, constants.emitterCandidates, candPos,
                                candNormal, rng);

        float W = reservoir.W;
        EmitterSample emitter = reservoir.sample;
        float3 toEmitter = emitter.position - candPos;
        float distanceSquared = dot(toEmitter, toEmitter);
        float3 w = toEmitter * rsqrt(distanceSquared);
        float cosSource = dot(candNormal, w);
        float cosEmitter = -dot(emitter.normal, w);
        bool visible = false;
        // The target floors the cosines (`emitterTarget`), so the kept sample
        // can face away; it then lights nothing and needs no ray.
        if (W > 0.0 && cosSource > 0.0 && cosEmitter > 0.0) {
            float3 radiance = cand.albedo / kPi * emitter.radiance
                * (cosSource * cosEmitter / distanceSquared * W);
            // From just off the source to just off the emitter's face, so
            // neither voxel blocks its own ray.
            float3 from = candPos + candNormal * 0.001;
            float3 to = emitter.position + emitter.normal * 0.001;
            RayDesc shadow;
            shadow.Origin = from;
            shadow.TMin = 0.0;
            shadow.TMax = length(to - from);
            shadow.Direction = (to - from) / shadow.TMax;
            visible = traceShadowRay(shadow, pl);
            if (visible) {
                params.sharc.addRadianceToSlot(pl.cacheSlot, radiance, float3(1.0, 1.0, 1.0));
            }
            if ((constants.flags & SHARC_FLAG_STATS) != 0) {
                float light = dot(radiance, kRec709Luminance);
                InterlockedAdd(constants.stats->emitterRays, 1u);
                InterlockedAdd(constants.stats->emitterLight, light);
                if (!visible) {
                    InterlockedAdd(constants.stats->emitterRaysBlocked, 1u);
                    InterlockedAdd(constants.stats->emitterLightBlocked, light);
                }
            }
        }
        if (first && reuse) {
            SharcReservoir next;
            next.sample = reservoir.sample.id;
            next.W = W;
            next.count = reservoir.count;
            params.sharc.keepReservoir(pl.cacheSlot, next);
        }
    }

    // ── One cosine-hemisphere bounce ─────────────────────────────────────
    // bounceMissShader (sky) or vox_sharc_update's closest-hit (cached terminator)
    // deposits the indirect contribution into pl.cacheSlot. The Lambertian
    // throughput (cand.albedo) is applied at the deposit site.
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

// Shadow miss — nothing blocks the ray to the sun or to the emitter. The
// ray-gen deposits the light.
[shader("miss")]
void shadowMissShader(inout SharcRayPayload payload) {
    payload.visible = 1;
}
