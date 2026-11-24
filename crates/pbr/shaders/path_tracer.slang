// The path tracer: textbook unidirectional path tracing of Lambertian and
// emissive surfaces, one path per pixel per frame. Every path starts at the
// surface the primary ray hit, read from the G-buffers, samples the sun at each
// surface it reaches, gathers whatever light its cosine-weighted bounces find,
// and ends by Russian roulette. Its light is added to the output, which holds
// the primary rays' sky and emission. All rays are traced from here, so
// recursion isn't needed.
//
// It's compiled three times: as is, the reference, and for the two passes that
// use the radiance cache (sharc.slang):
// - SHARC_UPDATE: a path for a pixel of every block adds the light it finds to
//   the cache, for each surface it reaches. Nothing is output.
// - SHARC_QUERY: a path ends at a surface it bounces to once the cache holds the
//   light that surface reflects.

import pbr_common;
import sky_atmosphere;
import random;
import path_tracer_common;
import sharc;

struct PathTracerPush {
    SharcConstants sharc;
    // Where SHARC_QUERY stores how many bounces each pixel's path took that it
    // didn't end at, row after row; null if they aren't wanted.
    uint *bounces;
    // Seeds the random numbers, with the pixel.
    uint frameIndex;
    // Bounces after the primary ray's hit; 0 leaves only direct light.
    uint maxBounces;
}
[vk::push_constant] uniform PathTracerPush pc;

static const float kPi = 3.14159265358979;

// The bounce from which Russian roulette may end a path.
static const uint kRouletteStart = 1;

// The cache stores the light a surface reflects divided by its albedo
// (SHARC_MATERIAL_DEMODULATION). A black surface reflects nothing, so it would
// add zeros and darken the other surfaces in its cell. The cache update therefore
// gives every surface at least this albedo.
static const float kMinCachedAlbedo = 1e-3;

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
#if SHARC_UPDATE
    uint2 screenSize;
    params.gbuffer.depthTexture.GetDimensions(screenSize.x, screenSize.y);
    // Inverting the frame index gives this pass different random numbers from the
    // pass that lights the frame, so that their samples are independent.
    uint seed = xxhash32(uint3(DispatchRaysIndex().xy, ~pc.frameIndex));
    uint2 pixel = DispatchRaysIndex().xy * pc.sharc.updateBlock;
    pixel += uint2(pcg(seed), pcg(seed)) % pc.sharc.updateBlock;
    if (any(pixel >= screenSize)) {
        return;
    }
#else
    uint2 screenSize = DispatchRaysDimensions().xy;
    uint2 pixel = DispatchRaysIndex().xy;
    uint seed = xxhash32(uint3(pixel, pc.frameIndex));
#endif

#if SHARC_UPDATE || SHARC_QUERY
    SharcParameters cache = pc.sharc.parameters();
#endif

    // The sky.
    if (params.gbuffer.depthTexture[pixel] == 0.0) {
        return;
    }

    // The surface the path is at. The output already has its emission.
    float3 position = params.pixelPositionWorldSpace(pixel, screenSize);
    float3 normal = params.gbuffer.normalTexture[pixel].xyz;
    float3 albedo = params.gbuffer.albedoSRGB[pixel].rgb;
    float3 emission = float3(0.0);

#if SHARC_UPDATE
    SharcState path;
    SharcInit(path);
#else
    float3 radiance = float3(0.0);
    float3 throughput = float3(1.0);
#endif
    uint bounce;
    for (bounce = 0;; bounce++) {
#if SHARC_UPDATE
        albedo = max(albedo, kMinCachedAlbedo);
#endif
        RayDesc ray;
        ray.Origin = offsetRayOrigin(position, normal);
        ray.TMin = 0.0;
        ray.TMax = params.uniforms.far;

        // The sunlight the surface reflects.
        float3 reflected = float3(0.0);
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
                reflected = albedo / kPi * cosSun
                          * params.sky.atmosphereParams.solar_illuminance
                          * params.sky.sampleTransmittanceAt(position - params.uniforms.cameraOrigin());
            }
        }
#if SHARC_UPDATE
        SharcHitData inCache = sharcSurface(cache, position, normal, albedo, min(emission, kMaxCachedRadiance));
        if (!SharcUpdateHit(cache, path, inCache, reflected, rand(seed))) {
            break;
        }
#else
        radiance += throughput * reflected;
#endif

        if (bounce == pc.maxBounces) {
            break;
        }
        ray.Direction = sampleCosineHemisphere(normal, seed);
#if SHARC_UPDATE
        SharcSetThroughput(path, albedo);
#else
        throughput *= albedo;
        if (bounce >= kRouletteStart) {
            float survival = min(max(throughput.r, max(throughput.g, throughput.b)), 0.95);
            if (rand(seed) >= survival) {
                break;
            }
            throughput /= survival;
        }
#endif

        PathTracerPayload surface;
        TraceRay(params.sceneBVH, RAY_FLAG_NONE, ~0, 0, 1, 0, ray, surface);
        if (surface.hitT < 0.0) {
            // Without the sun's disk, which every surface samples.
            float3 sky = params.sky.atmosphereParams.solar_illuminance
                       * params.sky.sampleScatteredTransfer(ray.Direction);
#if SHARC_UPDATE
            SharcUpdateMiss(cache, path, sky);
#else
            radiance += throughput * sky;
#endif
            break;
        }
        position = ray.Origin + ray.Direction * surface.hitT;
        normal = surface.normal;
        albedo = surface.albedo;
        emission = surface.emission;
#if SHARC_QUERY
        // A cell's average is too coarse for a surface nearer than the cell is
        // wide. Beyond that, a Lambertian surface's bounces spread far wider
        // than a cell.
        SharcHitData inCache = sharcSurface(cache, position, normal, albedo, emission);
        uint level = HashGridGetLevel(inCache.positionWorld, cache.hashGridParameters);
        float3 cached;
        if (
            surface.hitT > HashGridGetVoxelSize(level, cache.hashGridParameters)
            && SharcGetCachedRadiance(cache, inCache, cached, false)
        ) {
            // With the surface's emission.
            radiance += throughput * cached;
            break;
        }
#endif
#if !SHARC_UPDATE
        radiance += throughput * emission;
#endif
    }
#if SHARC_QUERY
    if (pc.bounces != nullptr) {
        pc.bounces[pixel.y * screenSize.x + pixel.x] = bounce;
    }
#endif
#if !SHARC_UPDATE
    params.outputTexture[pixel] += float4(radiance, 0.0);
#endif
}

[shader("miss")]
void missShader(inout PathTracerPayload payload) {
    payload.hitT = -1.0;
}
