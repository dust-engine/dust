struct Block
{
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

struct ShadowVoxParams {
    Block* geometry_info;
    uint64_t _pad0;
    uint64_t _pad1;
    float unit_size;
    uint32_t _pad2;

    Block getCurrentPrimitive() {
        return geometry_info[PrimitiveIndex()];
    }
};

float2 intersectAABB(float3 origin, float3 dir, float3 box_min, float3 box_max) {
    float3 tMin = (box_min - origin) / dir;
    float3 tMax = (box_max - origin) / dir;
    float3 t1 = min(tMin, tMax);
    float3 t2 = max(tMin, tMax);
    float t_min = max(max(t1.x, t1.y), t1.z);
    float t_max = min(min(t2.x, t2.y), t2.z);
    return float2(t_min, t_max);
}

uint8_t encode_index(uint8_t3 position) {
    return (position.x << 4) | (position.y << 2) | position.z;
}

/// Simplified DDA for shadow rays. No AMD workaround needed since hit attributes
/// are never read (closest hit shader is skipped for shadow rays).
void shadow_dda(float3 origin, float3 dir, uint64_t grid, float scale) {
    float2 t = intersectAABB(origin, dir, float3(0.0), float3(4.0));
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
        position += int8_t3(STEP * comp);
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

    ReportHit(hitDistance / scale, 0, uint8_t(0));
}

[shader("intersection")]
void main(uniform ShadowVoxParams params) {
    Block block = params.getCurrentPrimitive();
    float inv_unit = 1.0 / params.unit_size;
    shadow_dda(
        (ObjectRayOrigin() - block.coords * params.unit_size) * inv_unit,
        ObjectRayDirection() * inv_unit,
        block.mask,
        1.0
    );
}
