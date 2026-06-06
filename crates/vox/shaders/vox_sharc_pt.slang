// Voxel hit group for the SHARC Update path-tracer pipeline.
//
// The closest-hit returns hit data (worldPos via t, world-space face normal,
// albedo) through the payload; the ray-gen owns the SHARC state machine and
// the multi-bounce loop. The intersection shader mirrors the DDA in
// vox_pbr.slang — we keep it self-contained so this pipeline doesn't depend on
// G-buffer outputs.

import pbr_common;  // SharcHitPayload

struct Block {
    uint64_t mask;
    private uint32_t coords_packed;
    uint32_t material_ptr;

    property float3 coords {
        get {
            static const uint32_t mask = ((1 << 10) - 1);
            uint32_t3 min;
            min.x = coords_packed >> 20;
            min.y = (coords_packed >> 10) & mask;
            min.z = coords_packed & mask;
            return min << 2;
        }
    }
};

// 1 byte voxelId + 1 byte packed face normal — same encoding as vox_pbr.
struct HitAttribute {
    uint8_t voxelId;
    uint8_t normal;
    [mutating]
    void setNormal(int8_t3 n) {
        int8_t axis = (n.y & 1) * 2 + (n.z & 1) * 4;
        int8_t sign = ((n.x | n.y | n.z) >> 7) & 1;
        this.normal = axis + sign;
    }
    int8_t3 getNormal() {
        int axis = this.normal >> 1;
        int val = 1 - ((this.normal & 1) << 1);
        int8_t3 d;
        d.x = axis == 0 ? val : 0;
        d.y = axis == 1 ? val : 0;
        d.z = axis == 2 ? val : 0;
        return d;
    }
}

// AMD workaround: see vox_pbr.slang. AMD GPUs reflect the *last* intersection
// hitAttribute in closest-hit rather than the closest one; gate writes by
// `tHit <= RayTCurrent()` to keep behavior consistent.
__generic<A>
[ForceInline]
[require(glsl_hlsl_spirv, raytracing_intersection)]
bool ReportHitWithAMDWorkaround(float tHit, uint hitKind, A attributes) {
    __target_switch {
    case hlsl:
        __intrinsic_asm "ReportHit($0, $1, $2)";
    case glsl:
    case spirv:
        if (tHit <= RayTCurrent()) {
            [__vulkanHitAttributes]
            static A a;
            a = attributes;
        }
        return __reportIntersection(tHit, hitKind);
    }
}

struct VoxModelParams {
    Block* geometry_info;
    uint32_t* material_info;
    uint32_t* palette_info;
    float unit_size;
    uint32_t _pad;

    Block getCurrentPrimitive() { return geometry_info[PrimitiveIndex()]; }

    float4 getColorForVoxel(uint8_t voxelId) {
        if (!material_info || !palette_info) {
            return float4(1.0, 0.0, 1.0, 1.0);
        }
        Block currentPrimitive = getCurrentPrimitive();
        uint64_t masked = currentPrimitive.mask & ((uint64_t(1) << voxelId) - 1);
        uint32_t voxelMemoryOffset = countbits(masked);
        uint32_t paletteIndexPacked = material_info[(voxelMemoryOffset + currentPrimitive.material_ptr) / 4];
        uint8_t paletteIndex = (uint8_t)(paletteIndexPacked >> (((voxelMemoryOffset + currentPrimitive.material_ptr) & 3) * 8));
        uint8_t4 palette_color = bit_cast<uint8_t4>(palette_info[min(paletteIndex, 255)]);
        return float4(palette_color) / 255.0;
    }
}

[shader("closesthit")]
void closestHitShader(
    uniform VoxModelParams model,
    in HitAttribute hitAttr,
    inout SharcHitPayload payload,
) {
    float4 color = model.getColorForVoxel(hitAttr.voxelId);
    float3 normalObj = float3(hitAttr.getNormal());
    float3x4 o2w = ObjectToWorld3x4();
    float3 worldNormal =
        normalize(mul(float3x3(o2w[0].xyz, o2w[1].xyz, o2w[2].xyz), normalObj));

    payload.t = RayTCurrent();
    payload.albedo = color.xyz;
    payload.normal = worldNormal;
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
        (ObjectRayOrigin() - block.coords * params.unit_size) * inv_unit,
        ObjectRayDirection() * inv_unit,
        block.mask,
        1.0
    );
}
