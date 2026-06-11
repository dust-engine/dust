import pbr_common;
import vox_common;

float2 intersectAABB(float3 origin, float3 dir, float3 box_min, float3 box_max, out int8_t3 positionDelta) {
    float3 tMin = (box_min - origin) / dir;
    float3 tMax = (box_max - origin) / dir;
    float3 t1 = min(tMin, tMax);
    float3 t2 = max(tMin, tMax);
    float t_min = max(max(t1.x, t1.y), t1.z);
    float t_max = min(min(t2.x, t2.y), t2.z);
    float3 compResult = step(t1.yzx, t1.xyz) * step(t1.zxy, t1.xyz);
    positionDelta = int8_t3(sign(dir) * compResult);
    return float2(t_min, t_max);
}

uint8_t encode_index(uint8_t3 position) {
    return (position.x << 4) | (position.y << 2) | position.z;
}

// DDA mirrored from vox_pbr's dda(), trimmed of the visualization branches.
// Tracks `positionDelta` so we can hand the face normal to the closest hit
// shader for SHARC's worldNormal-keyed cache query.
void final_gather_dda(float3 origin, float3 dir, uint64_t grid, float scale) {
    int8_t3 positionDelta;
    float2 t = intersectAABB(origin, dir, float3(0.0), float3(4.0), positionDelta);
    if (t.x >= t.y || t.y <= 0.0)
        return;

    float hitDistance = max(t.x, RayTMin());
    float3 pos = origin + dir * hitDistance;
    int8_t3 position = int8_t3(clamp(int3(floor(pos)), 0, 3));

#define STEP sign(dir)
    float3 t_coef = 1.0 / dir;
    float3 t_bias = t_coef * origin;
    float3 tMax = (float3(position) + max(STEP, 0.0)) * t_coef - t_bias;
    float3 tDelta = float3(1.0) * t_coef * STEP;

    uint8_t hit = encode_index(position);
    while ((grid & (uint64_t(1) << hit)) == 0) {
        float3 comp = step(tMax.xyz, tMax.zxy) * step(tMax.xyz, tMax.yzx);
        positionDelta = int8_t3(STEP * comp);
        position += positionDelta;
        // Exit when the step pushed us out of [0,3]^3. Casting to uint8_t maps
        // any negative position (e.g. -1 → 255) above 3, so a single unsigned
        // compare catches both under- and over-flow.
        if (any(uint8_t3(position) > uint8_t(3))) {
            return;
        }
        hitDistance = min(min(tMax.x, tMax.y), tMax.z);
        tMax += tDelta * comp;
        hit = encode_index(position);
    }

    HitAttribute attr;
    attr.voxelId = hit;
    attr.setNormal(-positionDelta);
    ReportHitWithAMDWorkaround(hitDistance / scale, 0, attr);
}

[shader("intersection")]
void intersectionShader(uniform VoxModelParams params) {
    Block block = params.getCurrentPrimitive();
    float inv_unit = 1.0 / params.unit_size;
    final_gather_dda(
        (ObjectRayOrigin() - block.coords * params.unit_size) * inv_unit,
        ObjectRayDirection() * inv_unit,
        block.mask,
        1.0
    );
}

// Final-gather indirect via SHARC. The gather ray hits a surface here; we look
// up the cached outgoing radiance at that surface and additively fold it into
// the HDR, modulated by the *primary* surface's albedo (Lambertian throughput
// cancellation: (albedo / pi) * cos / pdf collapses to `primaryAlbedo` when
// the gather ray is cosine-importance-sampled — same pattern the miss shader
// uses for sky contribution).
//
// We ALWAYS push (worldPos, normal) into the SHARC candidate pool, hit or
// miss — not only on a miss. Only-on-miss seeding made each cell go dark
// before it could be re-added: once populated, a cell was never refreshed
// until Resolve evicted it as stale, at which point the query missed and it
// got re-seeded — a visible per-cell pulse. By re-seeding every frame the cell
// stays alive, and we weight the push by 1/(1+sampleNum) so under-sampled /
// freshly-seen cells still win pool slots over well-converged ones (whose
// low — but nonzero — weight only refreshes them occasionally). The
// primary-pixel luminance further modulates by visual impact: dark pixels
// (shadowed cavities) shouldn't outcompete bright ones for pool slots.
[shader("closesthit")]
void closestHitShader(uniform VoxModelParams model,
                      in HitAttribute hitAttr) {
    float3 worldPos = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    float3x4 o2w = ObjectToWorld3x4();
    float3 worldNormal = normalize(
        mul(float3x3(o2w[0].xyz, o2w[1].xyz, o2w[2].xyz), float3(hitAttr.getNormal())));

    uint2 pixel = DispatchRaysIndex().xy;
    float3 primaryAlbedo = params.gbuffer.albedoSRGB[pixel].rgb;

    // Query both the resolved radiance and the cell's current sample count.
    float3 cachedRadiance;
    float sampleNum;
    bool hit = params.sharc.getCachedRadiance(worldPos, worldNormal, cachedRadiance, sampleNum);

    // Fold the cached indirect into HDR on a usable hit — but not when a SHARC
    // debug view owns the HDR output (the cache-hit contribution must not touch
    // the visualized color; the cell is still re-seeded below).
    if (hit) {
        // TODO: premultiply the albedo to make this even faster.
        float4 current = params.outputTexture[pixel];
        params.outputTexture[pixel] =
            float4(current.rgb + primaryAlbedo * cachedRadiance, current.a);
    }

    // Re-seed this cell every frame, weighted by the inverse of its existing
    // sample count. Hit-point albedo (sampled from the voxel palette by this
    // same CHS) rides along in the candidate so the Update path next frame
    // doesn't need a probe ray to recover it.
    float3 hitAlbedo = model.getColorForVoxel(hitAttr.voxelId).rgb;
    float visualImpact = dot(primaryAlbedo, float3(0.2126, 0.7152, 0.0722));
    float weight = visualImpact / (1.0 + sampleNum);
    uint seed = pixel.x
                ^ (pixel.y * 0x100u)
                ^ (params.sharc.g_sharcConstants.frameIndex * 0x10000u);
    params.sharc.pushCandidate(worldPos, worldNormal, hitAlbedo, weight, seed);
}

