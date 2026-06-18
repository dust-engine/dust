import pbr_common;
import sky_atmosphere;

struct [raypayload] ShadowPayload {
    float shadow : read(caller) : write(caller, miss);
}

[shader("raygeneration")]
void main() {
    uint2 pixel = DispatchRaysIndex().xy;
    uint missShaderIndex = params.gbuffer.albedoUint[pixel].a;

    // albedoUint.a is a per-pixel miss shader index written by the primary pass:
    // real hits store 0; sky/UI store 0xFF as a "no surface" sentinel. Skip the
    // sentinel pixels; the rest is passed straight to TraceRay's MissShaderIndex.
    if (missShaderIndex == 0xFF) {
        return;
    }

    // Reconstruct world position from camera ray + hit distance
    float3 worldPos = params.pixelPositionWorldSpace(DispatchRaysIndex().xy, DispatchRaysDimensions().xy);
    float3 worldNormal = params.gbuffer.normalTexture[pixel].xyz;

    RayDesc ray;
    ray.Origin = worldPos + worldNormal * 0.01; // Offset along normal to avoid self-intersection
    ray.Direction = params.sky.atmosphereParams.sun_direction;
    ray.TMin = 0.0;
    ray.TMax = params.uniforms.far;

    ShadowPayload payload;
    payload.shadow = 0.0;
    TraceRay(
        params.sceneBVH,
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_FORCE_OPAQUE,
        ~0,
        0, 1, missShaderIndex,
        ray, payload
    );
}
