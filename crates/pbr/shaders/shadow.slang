import pbr_common;
import sky_atmosphere;
import random;

struct [raypayload] ShadowPayload {
    float shadow : read(caller) : write(caller, miss);
}

// Per-frame sampling seed for the sun-disk jitter. Uses the RayGen (0, 4)
// push-constant range that pbr.playout.ron declares (shared with the
// final-gather pass, which pushes its own frame index into the same range).
struct ShadowPush {
    uint frameIndex;
}
[vk::push_constant] uniform ShadowPush pc;

static const float kPi = 3.14159265358979;

// Branchless orthonormal basis around `n` (Duff et al., "Building an
// Orthonormal Basis, Revisited", JCGT 6.1, 2017).
void orthonormalBasis(float3 n, out float3 tangent, out float3 bitangent) {
    float s = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (s + n.z);
    float b = n.x * n.y * a;
    tangent = float3(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    bitangent = float3(b, s + n.y * n.y * a, -n.y);
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

    // Sample the shadow-ray direction uniformly within the sun's cone instead
    // of tracing the disk center: penumbra pixels then flicker between lit and
    // shadowed at the disk-visibility rate, which is exactly the stochastic
    // signal the temporal denoiser (MetalFX / DLSS-RR) reconstructs into a
    // soft edge. A single deterministic direction gives a hard aliased edge
    // the denoiser treats as noise and smears. The miss shader keeps NdotL and
    // transmittance at the mean sun direction — the disk only modulates
    // visibility, the standard estimator for a sub-degree light.
    uint seed = xxhash32(uint3(pixel.x, pixel.y, pc.frameIndex));
    float u0 = rand(seed);
    float u1 = rand(seed);
    float3 sunDir = params.sky.atmosphereParams.sun_direction;
    float cosThetaMax = cos(params.sky.atmosphereParams.sun_angular_radius);
    float cosTheta = 1.0 - u0 * (1.0 - cosThetaMax);
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    float phi = 2.0 * kPi * u1;
    float3 tangent, bitangent;
    orthonormalBasis(sunDir, tangent, bitangent);
    float3 sampleDir =
        normalize(sinTheta * (cos(phi) * tangent + sin(phi) * bitangent) + cosTheta * sunDir);

    RayDesc ray;
    ray.Origin = worldPos + worldNormal * 0.01; // Offset along normal to avoid self-intersection
    ray.Direction = sampleDir;
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
