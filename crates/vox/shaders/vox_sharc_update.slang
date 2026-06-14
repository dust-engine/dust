// Voxel hit group for the SHARC Update path-tracer pipeline.
//
// The closest-hit returns hit data (worldPos via t, world-space face normal,
// albedo) through the payload; the ray-gen owns the SHARC state machine and
// the multi-bounce loop. The intersection shader mirrors the DDA in
// vox_pbr.slang — we keep it self-contained so this pipeline doesn't depend on
// G-buffer outputs.

import pbr_common;
import vox_common;
import random;
import sharc;

[shader("closesthit")]
void closestHitShader(
    uniform VoxModelParams model,
    in HitAttribute hitAttr,
    inout SharcRayPayload payload,  // carries the source cell slot (set in ray-gen)
) {
    int8_t3 faceNormal = hitAttr.getNormal();
    float4 color = model.getColorForVoxel(hitAttr.voxelId);
    float3x4 o2w = ObjectToWorld3x4();
    float3 worldNormal =
        normalize(mul(float3x3(o2w[0].xyz, o2w[1].xyz, o2w[2].xyz), float3(faceNormal)));

    // Voxel-native SHARC cell key for the bounce target — same discrete
    // identity the visible-pass CHS uses (vox_common::voxelCellInfo), inlined
    // here since this pipeline keeps its own self-contained voxel structs.
    Block block = model.getCurrentPrimitive();
    float3 offsetInBox =
        float3(hitAttr.voxelId >> 4, (hitAttr.voxelId >> 2) & 0b11, hitAttr.voxelId & 0b11);
    int3 voxelCoord = int3(block.coords + offsetInBox);
    int axis = faceNormal.x != 0 ? 0 : (faceNormal.y != 0 ? 1 : 2);
    int sign = faceNormal[axis] < 0 ? 1 : 0;
    uint faceId = uint(axis * 2 + sign);
    // Voxel center (not face center) for the LOD — matches vox_common's
    // voxelCellInfo so the cascade and the visible query land on the same LOD.
    float3 centerObj = (float3(voxelCoord) + 0.5) * model.unit_size;
    float3 cellCenter = mul(o2w, float4(centerObj, 1.0));
    uint64_t bounceKey =
        params.sharc.cellKey(InstanceIndex(), voxelCoord, faceId, cellCenter);

    // The source cell — opened and counted once in the ray-gen — is addressed by
    // the slot carried in the payload, so no hash lookup here. The launch slot is
    // reloaded only to fetch the source albedo (throughput) on a cache hit and to
    // seed the cascade.
    uint launchSlot = DispatchRaysIndex().x;

    // Terminator: fold the bounce target's cached outgoing radiance into the
    // source cell, scaled by the source's Lambertian throughput. The source's
    // per-frame sample was already counted in the ray-gen, so a cache miss simply
    // deposits nothing.
    float3 cached;
    float bounceSampleNum;
    bool hasCache =
        params.sharc.getCachedRadianceByKey(bounceKey, cached, bounceSampleNum);
    if (hasCache) {
        float3 candAlbedo = params.sharc.candidateAt(launchSlot).albedo;
        params.sharc.addRadianceToSlot(payload.cacheSlot, cached, candAlbedo);
    }

    // Re-seed the bounce target into the candidate pool (cascade), weighted by
    // 1/(1+sampleNum) so under-converged cells keep priority. Hit-point albedo
    // rides along so next frame's Update needs no probe ray to recover it.
    float3 hitPos = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    float3 hitAlbedo = color.xyz;
    float w = dot(hitAlbedo, float3(0.2126, 0.7152, 0.0722)) / (1.0 + bounceSampleNum);
    uint seed = xxhash32(
        uint3(launchSlot, params.sharc.g_sharcConstants.frameIndex, 0xa5cc1c4du));
    params.sharc.pushCandidate(hitPos, worldNormal, hitAlbedo, bounceKey, w, seed);
}

// --- Intersection shader (verbatim DDA from vox_pbr.slang) ---

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

void dda(float3 origin, float3 dir, uint64_t grid, float scale) {
    int8_t3 positionDelta;
    float2 initialIntersectionT = intersectAABB(origin, dir, float3(0.0), float3(4.0), positionDelta);
    if (initialIntersectionT.x >= initialIntersectionT.y || initialIntersectionT.y <= 0.0) return;

    float hitDistance = max(initialIntersectionT.x, RayTMin());
    float3 ip = origin + dir * hitDistance;
    int8_t3 position = int8_t3(clamp(int3(floor(ip)), 0, 3));

#define STEP sign(dir)
    float3 t_coef = 1.0 / dir;
    float3 t_bias = t_coef * origin;
    float3 tMax = (float3(position) + max(STEP, 0.0)) * t_coef - t_bias;
    float3 tDelta = float3(1.0) * t_coef * STEP;

    uint8_t hitVoxelId = encode_index(position);
    while ((grid & (uint64_t(1) << hitVoxelId)) == 0) {
        float3 compResult = step(tMax.xyz, tMax.zxy) * step(tMax.xyz, tMax.yzx);
        positionDelta = int8_t3(STEP * compResult);
        position += positionDelta;
        if (any(uint8_t3(position) > uint8_t(3))) return;
        hitDistance = min(min(tMax.x, tMax.y), tMax.z);
        tMax += tDelta * compResult;
        hitVoxelId = encode_index(position);
    }

    HitAttribute hit;
    hit.voxelId = hitVoxelId;
    hit.setNormal(-positionDelta);
    ReportHitWithAMDWorkaround(hitDistance / scale, 0, hit);
}

[shader("intersection")]
void intersectionShader(uniform VoxModelParams params) {
    Block block = params.getCurrentPrimitive();
    float inv_unit = 1.0 / params.unit_size;
    dda(
        (ObjectRayOrigin() - float3(block.coords) * params.unit_size) * inv_unit,
        ObjectRayDirection() * inv_unit,
        block.mask,
        1.0
    );
}
