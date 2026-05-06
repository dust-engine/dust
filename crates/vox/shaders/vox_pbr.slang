import pbr_common;

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

// Usually one would expect the `hitAttributeEXT` value in the closest hit shader to be the one set
// during the intersection shader invocation of the closest hit point. This is the case on NVIDIA GPUs.
// However, on my AMD GPU (6600XT), the `hitAttributeEXT` value seems to reflect the value set during
// the most recent intersection shader invocation instead.
// Meaning that if additional intersection tests were performed after the closest hit, wrong values
// would be reflected in the closest hit shader.
// Here, for NVIDIA GPUs we always set the voxelId. On AMD GPUs we manually make the comparison with gl_RayTmaxEXT.
__generic<A>
[ForceInline]
[require(glsl_hlsl_spirv, raytracing_intersection)]
bool ReportHitWithAMDWorkaround(float tHit, uint hitKind, A attributes)
{
    __target_switch
    {
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

    Block getCurrentPrimitive() {
        return geometry_info[PrimitiveIndex()];
    }

    float4 getColorForVoxel(uint8_t voxelId) {
        if (!material_info || !palette_info) {
            return float4(1.0, 0.0, 1.0, 1.0);
        }
        Block currentPrimitive = getCurrentPrimitive();
        float3 offsetInBox = float3(voxelId >> 4, (voxelId >> 2) & 0b11, voxelId & 0b11);

        uint64_t masked = currentPrimitive.mask & ((uint64_t(1) << voxelId) - 1);
        uint32_t voxelMemoryOffset = countbits(masked);
        uint32_t paletteIndexPacked = material_info[(voxelMemoryOffset + currentPrimitive.material_ptr) / 4];
        uint8_t paletteIndex = (uint8_t)(paletteIndexPacked >> (((voxelMemoryOffset + currentPrimitive.material_ptr) & 3) * 8));

        // min(x, 255) is a NVIDIA workaround.
        uint8_t4 palette_color = bit_cast<uint8_t4>(palette_info[min(paletteIndex, 255)]);
        return float4(palette_color) / 255.0;
    }
}

float SRGBToLinear(float color) {
    return color < 0.04045 ? color / 12.92 : pow(abs(color + 0.055) / 1.055, 2.4);
}

[shader("closesthit")]
void closestHitShader(
    uniform VoxModelParams model,
    in uint8_t voxelId,
) {
    float4 color = model.getColorForVoxel(voxelId);                                // Colors sampled from the palette are sRGB
    params.gbuffer.albedoLinear[DispatchRaysIndex().xy] = float4(color.xyz, 1.0); //Directly write sRGB.

    // Compute face normal from voxel hit
    Block block = model.getCurrentPrimitive();
    float3 voxelPos = float3(voxelId >> 4, (voxelId >> 2) & 0x3, voxelId & 0x3);
    float3 hitPoint = ObjectRayOrigin() + ObjectRayDirection() * RayTCurrent();
    float3 voxelObjCorner = (block.coords + voxelPos) * model.unit_size;
    float3 hitLocal = (hitPoint - voxelObjCorner) / model.unit_size; // [0,1]^3 within the voxel

    float3 d0 = abs(hitLocal);
    float3 d1 = abs(hitLocal - 1.0);
    float3 minD = min(d0, d1);
    float3 normalSign = select(d0 < d1, float3(-1.0), float3(1.0));

    float3 normal;
    if (minD.x < minD.y && minD.x < minD.z) {
        normal = float3(normalSign.x, 0, 0);
    } else if (minD.y < minD.z) {
        normal = float3(0, normalSign.y, 0);
    } else {
        normal = float3(0, 0, normalSign.z);
    }

    // Transform to world space
    float3x4 o2w = ObjectToWorld3x4();
    float3 worldNormal = normalize(mul(float3x3(o2w[0].xyz, o2w[1].xyz, o2w[2].xyz), normal));
    params.gbuffer.normalTexture[DispatchRaysIndex().xy] = float4(worldNormal, 1.0);

    // Encode reverse-infinite-Z hardware depth: `near / linearViewZ`, where
    // linearViewZ is the perpendicular distance from the camera plane to the
    // hit (ray length × cos(angle_from_forward)). Sky/UI write 0 = infinity.
    float3 dirCam = params.uniforms.primaryRayDirCameraSpace(DispatchRaysIndex().xy, DispatchRaysDimensions().xy);
    float linearViewZ = RayTCurrent() * -dirCam.z;
    params.gbuffer.depthTexture[DispatchRaysIndex().xy] = params.uniforms.near / linearViewZ;

    // Screen-space motion vector: take the object-space hit point, transform
    // it through the instance's previous-frame world-from-object matrix, then
    // project through last frame's camera. This captures both camera motion
    // and per-object motion for moving instances.
    float3 objectHit = ObjectRayOrigin() + ObjectRayDirection() * RayTCurrent();
    uint2 screenSize = DispatchRaysDimensions().xy;
    float2 prevPixel = params.prevPixelFromObjectHit(objectHit, InstanceIndex(), screenSize);
    // Subtract jitter: the jittered ray sampled the world at sub-pixel
    // p - jitter in unjittered NDC, so the un-jittered current projection
    // of this hit is p - jitter. DLSS expects MV in unjittered grid.
    float2 currPixel = float2(DispatchRaysIndex().xy) - float2(params.uniforms.jitter_x, params.uniforms.jitter_y);
    params.gbuffer.motionVectorTexture[DispatchRaysIndex().xy] = prevPixel - currPixel;
}

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

/**
  Intersects ray with a unix box with 4x4x4 voxels. A branchless implementation of
  Amanatides, John & Woo, Andrew. (1987). A Fast Voxel Traversal Algorithm for Ray Tracing. Proceedings of EuroGraphics. 87.

  @param(origin): The origin of the ray.
  @param(dir):    The direction of the ray. Does not have to be normalized.
  @param(grid):   The occupancy of the 64 voxels encoded in z-curve order.
*/
void dda(float3 origin, float3 dir, uint64_t grid, float scale) {
    // Init phase

    // The initialization phase begins by identifying the voxel in which the ray origin is found. If the ray
    // origin is outside the grid, we find the point in which the ray enters the grid and take the adjacent voxel.
    // The integer variables `position` are initialized to the starting voxel coordinates.

    // We assume that the AABB box is located in 0-1. We extend that to 0-4 so we match our DDAed unit box of 4x4x4.
    float2 initialIntersectionT = intersectAABB(origin, dir, float3(0.0, 0.0, 0.0), float3(4.0, 4.0, 4.0));
    if (initialIntersectionT.x >= initialIntersectionT.y) {
        // No intersection
        return;
    }
    if (initialIntersectionT.y <= 0.0) {
        return;
    }

    #ifdef VISUALIZE_BIG_BLOCKS
    if (initialIntersectionT.x >= initialIntersectionT.y) {
        // No intersection
        return;
    }
    if (initialIntersectionT.y <= 0.0) {
        return;
    }
    if (grid != 0) {
        ReportHitWithAMDWorkaround(initialIntersectionT.x, 0, 0xF);
    }
    #endif

    float hitDistance = max(initialIntersectionT.x, RayTMin());
    float3 initialIntersectionPoint = origin + dir * hitDistance;

    int8_t3 position = int8_t3(clamp(int3(floor(initialIntersectionPoint)), 0, 3));

// In  addition, the variables stepX and stepY are initialized to either 1 or -1 indicating whether X and Y
// are incremented or decremented as theray crosses voxel boundaries (this is determined by the sign of the
// x and y components of v)
#define STEP sign(dir)

    // Next, we determine the value of t at which the ray crosses the first vertical voxel boundary and
    // store it in variable tMaxX. We perform a similar computation in y and store the result in tMaxY. The
    // minimum of these two values will indicate how much we can travel along the ray and still remain in the
    // current voxel.

    // Neo's note: Define our ray as v = origin + t * dir, so t = (v - origin) / dir = v / dir - origin / dir.
    // We assign t_coef = 1 / dir and t_bias = origin / dir such that t(v) = v * t_coef - t_bias.
    float3 t_coef = 1.0 / dir;
    float3 t_bias = t_coef * origin;

    float3 tMax = (float3(position) + max(STEP, 0.0)) * t_coef - t_bias;

    // Finally, we compute tDeltaX and tDeltaY. TDeltaX indicates how far along the ray we must move
    // (in units of t) for the horizontal component of such a movement to equal the width of a voxel. Similarly,
    // we store in tDeltaY the amount of movement along the ray which has a vertical component equal to the
    // height of a voxel.
    float3 tDelta = float3(1.0, 1.0, 1.0) * t_coef * STEP;

    // Intersection phase
    uint8_t hit = encode_index(position);

    while ((grid & (uint64_t(1) << hit)) == 0) {
        float3 compResult = step(tMax.xyz, tMax.zxy) * step(tMax.xyz, tMax.yzx);
        int8_t3 positionDelta = int8_t3(STEP * compResult); // Either +1 or -1 or 0 on each digit.
        position += positionDelta;

        hitDistance = min(min(tMax.x, tMax.y), tMax.z);
        if (hitDistance + 0.001 >= initialIntersectionT.y) {
            return;
        }
        tMax += tDelta * compResult;

        hit = encode_index(position);
    };

    float realT = hitDistance / scale;

    ReportHitWithAMDWorkaround(realT, 0, hit);
}

[shader("intersection")]
void intersectionShader(
    uniform VoxModelParams params,
) {
    Block block = params.getCurrentPrimitive();

    // Object-space leaf corner is block.coords * unit_size. Map ray into
    // voxel-space ([0,4]^3 with unit-sized cells) by subtracting the leaf
    // corner and dividing by unit_size. Both origin offset and direction
    // are scaled by 1/unit_size, so dda's t parameter equals object-space t.
    float inv_unit = 1.0 / params.unit_size;
    dda(
        (ObjectRayOrigin() - block.coords * params.unit_size) * inv_unit,
        ObjectRayDirection() * inv_unit,
        block.mask,
        1.0
    );
}
