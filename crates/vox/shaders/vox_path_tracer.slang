import vox_common;
import path_tracer_common;

// The voxel's face as the path tracer sees it: a Lambertian surface that may
// emit. The hit group shares the primary rays' intersection shader.
[shader("closesthit")]
void closestHitShader(
    uniform VoxModelParams model,
    inout PathTracerPayload payload,
    in HitAttribute hitAttr,
) {
    VoxMaterial mat = model.getMaterialForVoxel(hitAttr.voxelId);
    float3 color = mat.albedo().rgb; // sRGB-encoded

    float3x4 o2w = ObjectToWorld3x4();
    float3x3 rotation = float3x3(o2w[0].xyz, o2w[1].xyz, o2w[2].xyz);

    payload.hitT = RayTCurrent();
    payload.normal = normalize(mul(rotation, float3(hitAttr.getNormal())));
    payload.albedo = select(color <= 0.04045, color / 12.92, pow((color + 0.055) / 1.055, 2.4));
    // As the primary rays output it.
    payload.emission = color * mat.emission;
}
