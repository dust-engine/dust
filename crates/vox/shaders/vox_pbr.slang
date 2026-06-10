import pbr_common;
import vox_common;



[shader("closesthit")]
void closestHitShader(
    uniform VoxModelParams model,
    in HitAttribute hitAttr,
) {
    float4 color = model.getColorForVoxel(hitAttr.voxelId);                                // Colors sampled from the palette are sRGB
    params.gbuffer.albedoLinear[DispatchRaysIndex().xy] = float4(color.xyz, 1.0); //Directly write sRGB.

    // The face the ray entered is whichever axis the dda last stepped across;
    // the normal points opposite the ray direction on that axis.
    float3 normal = float3(hitAttr.getNormal());

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
    int8_t3 positionDelta;
    float2 initialIntersectionT = intersectAABB(origin, dir, float3(0.0, 0.0, 0.0), float3(4.0, 4.0, 4.0), positionDelta);
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
    uint8_t hitVoxelId = encode_index(position);

    while ((grid & (uint64_t(1) << hitVoxelId)) == 0) {
        float3 compResult = step(tMax.xyz, tMax.zxy) * step(tMax.xyz, tMax.yzx);
        // The axis selected is the one whose tMax was smallest. Either +1 or -1 on that position.
        // 0 on the other two positions.
        positionDelta = int8_t3(STEP * compResult); 
        position += positionDelta;

        // Exit when the step pushed us out of [0,3]^3. Casting to uint8_t maps
        // any negative position (e.g. -1 → 255) above 3, so a single unsigned
        // compare catches both under- and over-flow.
        if (any(uint8_t3(position) > uint8_t(3))) {
            return;
        }
        hitDistance = min(min(tMax.x, tMax.y), tMax.z);
        tMax += tDelta * compResult;

        hitVoxelId = encode_index(position);
    };

    HitAttribute hit;
    hit.voxelId = hitVoxelId;
    hit.setNormal(-positionDelta);
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
