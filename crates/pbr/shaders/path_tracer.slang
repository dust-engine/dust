// The reference path tracer: textbook unidirectional path tracing of Lambertian
// and emissive surfaces, one path per pixel per frame. Every path starts at the
// surface the primary ray hit, read from the G-buffers, samples the sun at each
// surface it reaches, gathers whatever light its cosine-weighted bounces find,
// and ends by Russian roulette. Its light is added to the output, which holds
// the primary rays' sky and emission. All rays are traced from here, so
// recursion isn't needed.

import pbr_common;
import sky_atmosphere;
import random;
import path_tracer_common;

struct PathTracerPush {
    // Seeds the random numbers, with the pixel.
    uint frameIndex;
    // Bounces after the primary ray's hit; 0 leaves only direct light.
    uint maxBounces;
}
[vk::push_constant] uniform PathTracerPush pc;

static const float kPi = 3.14159265358979;

// The bounce from which Russian roulette may end a path.
static const uint kRouletteStart = 1;

// Branchless orthonormal basis around `n` (Duff et al., "Building an
// Orthonormal Basis, Revisited", JCGT 6.1, 2017).
void orthonormalBasis(float3 n, out float3 tangent, out float3 bitangent) {
    float s = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (s + n.z);
    float b = n.x * n.y * a;
    tangent = float3(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    bitangent = float3(b, s + n.y * n.y * a, -n.y);
}

// Cosine-weighted direction around `n`. A Lambertian bounce is sampled in
// proportion to its BRDF times the cosine, so its weight is just the albedo.
float3 sampleCosineHemisphere(float3 n, inout uint seed) {
    float u0 = rand(seed);
    float phi = 2.0 * kPi * rand(seed);
    float r = sqrt(u0);
    float3 tangent, bitangent;
    orthonormalBasis(n, tangent, bitangent);
    return normalize(r * cos(phi) * tangent + r * sin(phi) * bitangent + sqrt(1.0 - u0) * n);
}

// Uniform direction over the sun's disk.
float3 sampleSunDisk(inout uint seed) {
    // `d` is 1 - cos of the angle from the disk's center. Its largest value
    // comes from the half angle: 1 - cos(radius) would keep few of its bits.
    float sinHalf = sin(0.5 * params.sky.atmosphereParams.sun_angular_radius);
    float d = rand(seed) * 2.0 * sinHalf * sinHalf;
    float sinTheta = sqrt(d * (2.0 - d));
    float phi = 2.0 * kPi * rand(seed);
    float3 sun = params.sky.atmosphereParams.sun_direction;
    float3 tangent, bitangent;
    orthonormalBasis(sun, tangent, bitangent);
    return normalize(sinTheta * (cos(phi) * tangent + sin(phi) * bitangent) + (1.0 - d) * sun);
}

// Moves a ray's origin off the surface at `p`, far enough that rounding can't
// put it on the other side (Wächter and Binder, "A Fast and Robust Method for
// Avoiding Self-Intersection", Ray Tracing Gems, 2019).
float3 offsetRayOrigin(float3 p, float3 n) {
    static const float kOrigin = 1.0 / 32.0;
    static const float kFloatScale = 1.0 / 65536.0;
    static const float kIntScale = 256.0;
    int3 offset = int3(kIntScale * n);
    float3 moved = asfloat(asint(p) + select(p < 0.0, -offset, offset));
    return select(abs(p) < kOrigin, p + kFloatScale * n, moved);
}

[shader("raygeneration")]
void main() {
    uint2 pixel = DispatchRaysIndex().xy;
    // The sky.
    if (params.gbuffer.depthTexture[pixel] == 0.0) {
        return;
    }
    uint seed = xxhash32(uint3(pixel, pc.frameIndex));

    // The surface the path is at.
    float3 position = params.pixelPositionWorldSpace(pixel, DispatchRaysDimensions().xy);
    float3 normal = params.gbuffer.normalTexture[pixel].xyz;
    float3 albedo = params.gbuffer.albedoSRGB[pixel].rgb;

    float3 radiance = float3(0.0);
    float3 throughput = float3(1.0);
    for (uint bounce = 0;; bounce++) {
        RayDesc ray;
        ray.Origin = offsetRayOrigin(position, normal);
        ray.TMin = 0.0;
        ray.TMax = params.uniforms.far;

        ray.Direction = sampleSunDisk(seed);
        float cosSun = dot(normal, ray.Direction);
        if (cosSun > 0.0) {
            PathTracerPayload sun;
            sun.hitT = 0.0;
            TraceRay(
                params.sceneBVH,
                RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
                ~0, 0, 1, 0, ray, sun
            );
            if (sun.hitT < 0.0) {
                radiance += throughput * albedo / kPi * cosSun
                          * params.sky.atmosphereParams.solar_illuminance
                          * params.sky.sampleTransmittanceAt(position - params.uniforms.cameraOrigin());
            }
        }

        if (bounce == pc.maxBounces) {
            break;
        }
        ray.Direction = sampleCosineHemisphere(normal, seed);
        throughput *= albedo;
        if (bounce >= kRouletteStart) {
            float survival = min(max(throughput.r, max(throughput.g, throughput.b)), 0.95);
            if (rand(seed) >= survival) {
                break;
            }
            throughput /= survival;
        }

        PathTracerPayload surface;
        TraceRay(params.sceneBVH, RAY_FLAG_NONE, ~0, 0, 1, 0, ray, surface);
        if (surface.hitT < 0.0) {
            // Without the sun's disk, which every surface samples.
            radiance += throughput * params.sky.atmosphereParams.solar_illuminance
                      * params.sky.sampleScatteredTransfer(ray.Direction);
            break;
        }
        radiance += throughput * surface.emission;
        position = ray.Origin + ray.Direction * surface.hitT;
        normal = surface.normal;
        albedo = surface.albedo;
    }
    params.outputTexture[pixel] += float4(radiance, 0.0);
}

[shader("miss")]
void missShader(inout PathTracerPayload payload) {
    payload.hitT = -1.0;
}
