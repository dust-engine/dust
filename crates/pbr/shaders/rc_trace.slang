// Radiance cascades: traces this frame's rays and splits each one into the
// intervals of its cascades (§5, step 4 in rc_common.slang).

import pbr_common;
import rc_common;
import random;

static const float kPi = 3.14159265358979;
static const float RC_MAX_RADIANCE = 30000.0;

// Radiance leaving a point a ray hit, toward the ray: emission, the sun, and if
// there's a secondary cache, the light it holds for that point.
float3 shadeHit(float3 position, RcPayload hit) {
    float3 radiance = hit.emission;
    float3 sunDir = params.sky.atmosphereParams.sun_direction;
    float NdotL = dot(hit.normal, sunDir);
    if (NdotL > 0.0) {
        RayDesc shadow;
        shadow.Origin = position + hit.normal * 0.01;
        shadow.Direction = sunDir;
        shadow.TMin = 0.0;
        shadow.TMax = params.uniforms.far;
        RcPayload blocked;
        blocked.t = 0.0;
        TraceRay(
            params.sceneBVH,
            RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_FORCE_OPAQUE,
            ~0, 0, 1, 0, shadow, blocked
        );
        if (blocked.t < 0.0) {
            float3 transmittance = params.sky.sampleTransmittanceAt(position - params.uniforms.cameraOrigin());
            radiance += hit.albedo / kPi * params.sky.atmosphereParams.solar_illuminance * transmittance * NdotL;
        }
    }
    if ((params.rc.flags & RC_FLAG_BOUNCE) != 0) {
        // Lambertian: albedo / π times irradiance, π times mean radiance.
        uint lod = rcLodOf(position).level + params.rc.bounce.lodOffset;
        float4 irradiance = rcSampleIrradiance(
            params.rc.bounce, params.rcBounceAtlas, lod, rcGridPoint(position, hit.normal, lod), hit.normal);
        radiance += hit.albedo * rcUnpremultiply(irradiance);
    }
    return radiance;
}

// The ray's contribution to each cascade up to the one its hit falls in, whose
// intervals are 4 times longer per cascade and 2 times longer per LOD (Eq. 4).
// Nothing beyond: the higher cascades learn nothing from a ray that stopped short
// of them, and treating it as a hit there biases them toward short rays (§5).
void deposit(uint probe, uint lod, uint direction, float t, float3 radiance) {
    float intervals = t / (params.rc.baseRayLength * exp2(float(lod)));
    uint hitCascade = uint(clamp(ceil(log2(max(intervals, 1.0)) * 0.5), 0.0, float(RC_CASCADES - 1u)));
    for (uint n = 0; n <= hitCascade && probe != RC_INVALID; n++) {
        RcCascade c = params.rc.cascades[n];
        uint index = c.dataOffset + probe * rcDirectionCount(n)
            + (direction >> (2u * (RC_CASCADES - 1u - n)));
        if (n == hitCascade) {
            if (radiance.r > 0.0) {
                InterlockedAdd(params.rc.accum[index].r, radiance.r);
            }
            if (radiance.g > 0.0) {
                InterlockedAdd(params.rc.accum[index].g, radiance.g);
            }
            if (radiance.b > 0.0) {
                InterlockedAdd(params.rc.accum[index].b, radiance.b);
            }
            InterlockedAdd(params.rc.accum[index].counts, 1u);
        } else {
            InterlockedAdd(params.rc.accum[index].counts, 0x10001u);
        }
        probe = params.rc.probes[c.probeOffset + probe].parent;
    }
}

[shader("raygeneration")]
void rcRayGen() {
    bool secondary = (params.rc.flags & RC_FLAG_SECONDARY) != 0;
    float3 position;
    float3 normal;
    RcLod lod;
    uint level;
    if (secondary) {
        RcHit source = params.rc.hits[DispatchRaysIndex().x];
        position = source.position;
        normal = rcDecodeNormal(source.normal);
        lod = rcLodOf(position);
        level = lod.level + params.rc.lodOffset;
        lod.blend = 0.0;
    } else {
        uint2 pixel = DispatchRaysIndex().xy;
        if (!rcIsSurfacePixel(pixel)) {
            return;
        }
        position = params.pixelPositionWorldSpace(pixel, params.rc.renderExtent);
        normal = params.gbuffer.normalTexture[pixel].xyz;
        lod = rcLodOf(position);
        level = lod.level + params.rc.lodOffset;
        if (!rcPixelTracesRay(pixel, position, level)) {
            return;
        }
    }

    float3 gridPoint = rcGridPoint(position, normal, level);
    uint probe = rcFindProbe(0, rcKey(level, rcCellOf(gridPoint, level, 0)));
    if (probe == RC_INVALID) {
        return;
    }

    // The next point of this probe's stretch of the R2 sequence, offset by a
    // per-frame amount (Algorithm 3). Directions pointing into the surface flip
    // to their opposite.
    uint sequence;
    InterlockedAdd(params.rc.probes[params.rc.cascades[0].probeOffset + probe].rayCursor, 1u, sequence);
    float2 uv = frac(rcR2(sequence) + params.rc.jitter);
    float3 direction = rcDecodeDirection(uv);
    if (dot(direction, normal) < 0.0) {
        direction = -direction;
        uv = float2(frac(uv.x + 0.5), 1.0 - uv.y);
    }

    RayDesc ray;
    ray.Origin = position + normal * 0.01;
    ray.Direction = direction;
    ray.TMin = 0.0;
    ray.TMax = params.uniforms.far;
    RcPayload hit;
    hit.t = -1.0;
    TraceRay(params.sceneBVH, RAY_FLAG_FORCE_OPAQUE, ~0, 0, 1, 0, ray, hit);

    float t;
    float3 radiance;
    if (hit.t >= 0.0) {
        t = hit.t;
        float3 hitPosition = ray.Origin + direction * t;
        radiance = shadeHit(hitPosition, hit);
        if ((params.rc.flags & RC_FLAG_RECORD_HITS) != 0
            && float(xxhash32(uint3(probe, sequence, params.rc.frameIndex))) * (1.0 / 4294967296.0) < params.rc.bounceRayFraction) {
            uint index;
            InterlockedAdd(params.rc.hitCounters[RC_HIT_COUNT], 1u, index);
            if (index < params.rc.hitCapacity) {
                RcHit record;
                record.position = hitPosition;
                record.normal = rcEncodeNormal(hit.normal);
                params.rc.hits[index] = record;
            }
        }
    } else {
        // Past the top cascade's interval there's only the sky, which the last
        // cascade takes as hit (§4). The sun's disk is the direct light's.
        t = params.uniforms.far;
        radiance = params.sky.atmosphereParams.solar_illuminance * params.sky.sampleScatteredTransfer(direction);
    }

    // A value the 16-bit history can't hold would stay in it, and spread to every
    // probe merging from it. Comparing with 0 also catches NaN.
    radiance = select(radiance > 0.0, min(radiance, RC_MAX_RADIANCE), 0.0);

    uint directionIndex = rcDirectionIndex(uv);
    deposit(probe, level, directionIndex, t, radiance);
    if (lod.blend > 0.0) {
        uint coarser = level + 1u;
        float3 coarserPoint = rcGridPoint(position, normal, coarser);
        deposit(rcFindProbe(0, rcKey(coarser, rcCellOf(coarserPoint, coarser, 0))), coarser, directionIndex, t, radiance);
    }
}

// Both the cache rays' and the shadow rays'.
[shader("miss")]
void rcMiss(inout RcPayload payload) {
    payload.t = -1.0;
}
