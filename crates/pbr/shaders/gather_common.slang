module gather_common;
import pbr_common;
import random;

// The final-gather ray of `pixel` this frame: a cosine-weighted hemisphere
// sample around `N`, seeded per pixel and frame. The screen trace and the
// hardware fallback both call this so a ray resumed from the last
// screen-visible point continues in exactly the direction the screen trace
// followed.
public float3 gatherRayDirection(uint2 pixel, float3 N, uint frameIndex) {
    uint seed = xxhash32(uint3(pixel.x, pixel.y, frameIndex));
    float u0 = rand(seed);
    float u1 = rand(seed);
    float r = sqrt(u0);
    float phi = 6.2831853071795864 * u1;
    float3 tangentDir = float3(r * cos(phi), r * sin(phi), sqrt(1.0 - u0));

    float3 T, B;
    if (abs(N.y) < 0.999) {
        T = normalize(cross(N, float3(0, 1, 0)));
    } else {
        T = normalize(cross(N, float3(1, 0, 0)));
    }
    B = cross(N, T);
    return normalize(T * tangentDir.x + B * tangentDir.y + N * tangentDir.z);
}
