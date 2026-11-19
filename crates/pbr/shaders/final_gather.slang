import pbr_common;
import gather_common;

// Push constant for the final-gather ray-gen: the per-frame sampling seed.
// The same value rides in `ScreenTraceConstants::frameIndex` for the screen
// trace, so both passes sample the same direction per pixel.
struct FinalGatherPush {
    uint frameIndex;
}
[vk::push_constant] uniform FinalGatherPush pc;

// Hardware fallback of the screen-space gather (screen_trace.slang). With
// SCREEN_TRACE_FLAG_ENABLED the dispatch is 1D over the compacted list of
// pixels whose screen trace missed (sized by the GPU count through
// trace_rays_indirect), and each ray resumes just before the last point the
// screen trace could still see - Lumen's pullback
// (LumenScreenProbeHardwareRayTracing.usf). With screen tracing off the
// dispatch covers the screen and every surface pixel traces from the surface.
[shader("raygeneration")]
void finalGatherRayGenShader() {
    ScreenTraceConstants st = params.screenTrace;
    uint2 screenSize = uint2(st.renderExtent);
    bool screenTraced = (st.flags & SCREEN_TRACE_FLAG_ENABLED) != 0;

    uint2 pixel;
    if (screenTraced) {
        uint packed = params.compactedTraceTexels[DispatchRaysIndex().x];
        pixel = uint2(packed & 0xFFFF, packed >> 16);
    } else {
        pixel = DispatchRaysIndex().xy;
        // albedoUint.a is a per-pixel miss shader index written by the primary
        // pass: real hits store 0; sky/UI store 0xFF as a "no surface"
        // sentinel.
        if (params.gbuffer.albedoUint[pixel].a == 0xFF) {
            return;
        }
    }
    uint missShaderIndex = params.gbuffer.albedoUint[pixel].a;

    float3 worldPos = params.pixelPositionWorldSpace(pixel, screenSize);
    float3 N = params.gbuffer.normalTexture[pixel].xyz;
    float3 worldDir = gatherRayDirection(pixel, N, pc.frameIndex);

    // Resume where the screen trace stopped seeing the ray, backed up by the
    // pullback so a surface the trace grazed is not skipped.
    float tMin = 0.0;
    if (screenTraced) {
        tMin = max(decodeTraceHit(params.traceHit[pixel]).distance - st.pullbackBias, 0.0);
    }

    RayDesc ray;
    ray.Origin = worldPos + N * 0.01;
    ray.Direction = worldDir;
    ray.TMin = tMin;
    ray.TMax = params.uniforms.far;

    FinalGatherPayload payload;
    payload.pixel = pixel;
    TraceRay(
        params.sceneBVH,
        RAY_FLAG_FORCE_OPAQUE,
        ~0,
        0, 1, missShaderIndex,
        ray, payload
    );
}

[shader("miss")]
void finalGatherMissShader(inout FinalGatherPayload payload) {
    uint2 pixel = payload.pixel;
    float4 current = params.outputTexture[pixel];

    float3 worldDir = WorldRayDirection();
    float3 skyLuminance = params.sky.atmosphereParams.solar_illuminance
        * params.sky.sampleScatteredTransfer(worldDir);

    // Cosine-weighted importance sampling of Lambertian BRDF:
    //   L_indirect = albedo * (1/π) * L_i * cos(θ) / pdf = albedo * L_i
    // The 1/π (BRDF) and cos(θ) (integrand) cancel against pdf(ω) = cos(θ)/π.
    float3 albedo = params.gbuffer.albedoSRGB[pixel].rgb;
    params.outputTexture[pixel] = float4(current.rgb + albedo * skyLuminance, current.a);
}
