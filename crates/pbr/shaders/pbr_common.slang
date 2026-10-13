import sky_atmosphere;
module pbr_common;
__exported import sharc;

public struct SkyAtmosphereParams {
[Stages(STAGE_RAYGEN | STAGE_MISS | STAGE_COMPUTE)]
    public ConstantBuffer<AtmosphereParams> atmosphereParams;
[Stages(STAGE_RAYGEN | STAGE_MISS | STAGE_COMPUTE)]
    public Texture2D<float4> transmittanceLUT;
[Stages(STAGE_RAYGEN | STAGE_MISS | STAGE_COMPUTE)]
    public Texture2D<float4> skyViewLUT;
[Stages(STAGE_RAYGEN | STAGE_MISS | STAGE_COMPUTE)]
    public SamplerState linearSampler;

    // Sample the atmosphere's scattered-light contribution along `worldDir` from
    // the SkyView LUT
    //
    // Returns: unitless transfer factor. To get luminance [cd/m²] the caller must
    // multiply by solar_illuminance [lux]. This is because the LUT-generation
    // pass (sky_view_lut.slang) folds atmosphereParams.solar_illuminance into the
    // in-scattering integral, and we keep solar_illuminance normalized to (1,1,1)
    // so the LUT stores a pure transfer. See AtmosphereParams::solar_illuminance.
    //
    // Assumes `worldDir` is unit-length. Returns 0 if the camera is above the top
    // of the atmosphere (aerial-perspective path not yet implemented).
    public float3 sampleScatteredTransfer(float3 worldDir) {
        // Atmosphere frame: planet-centered, +Y up, so world origin sits at
        // altitude=bottom_radius above the planet center (see camera_position doc).
        float3 worldPos = atmosphereParams.camera_position + float3(0.0, atmosphereParams.bottom_radius, 0.0);
        float viewHeight = max(length(worldPos), atmosphereParams.bottom_radius + 0.001);;

        float3 upVector = worldPos / viewHeight;
        float viewZenithCosAngle = dot(worldDir, upVector);

        // Check if ray intersects ground
        float3 earthO = float3(0.0, 0.0, 0.0);
        bool intersectsGround = raySphereIntersectNearest(worldPos, worldDir, earthO, atmosphereParams.bottom_radius) >= 0.0;

        // Sky rendering using Sky View LUT
        if (viewHeight < atmosphereParams.top_radius) {
            float3 sideVector = normalize(cross(upVector, worldDir));
            float3 forwardVector = normalize(cross(sideVector, upVector));
            float2 lightOnPlane = float2(dot(atmosphereParams.sun_direction, forwardVector), dot(atmosphereParams.sun_direction, sideVector));
            lightOnPlane = normalize(lightOnPlane);
            float lightViewCosAngle = lightOnPlane.x;

            float2 skyUV;
            skyViewParamsToUv(atmosphereParams, intersectsGround, viewZenithCosAngle, lightViewCosAngle, viewHeight, skyUV);

            return skyViewLUT.SampleLevel(linearSampler, skyUV, 0).rgb;
        }

        // TODO: aerial perspective
        return float3(0.0, 0.0, 0.0);
    }

    // Direct sun-disk luminance [cd/m²] along `worldDir`. Returns 0 unless
    // `worldDir` is inside the sun's angular radius and the sun is not occluded
    // by the planet.
    public float3 getSunLuminance(float3 worldDir) {
        float3 worldPos = atmosphereParams.camera_position + float3(0.0, atmosphereParams.bottom_radius, 0.0);
        return atmosphereParams.getSunLuminance(worldPos, worldDir);
    }

    // Transmittance from a surface point toward the sun, sampled from the
    // transmittance LUT. `samplePosFromCamera` is the sample point's location
    // in world-oriented axes with the camera as origin (i.e. worldPos -
    // cameraOrigin). World axes must share orientation with atmosphere-space
    // axes (+Y up); this holds in Dust because the game world is a flat plane
    // tangent to the atmosphere's curved earth.
    public float3 sampleTransmittanceAt(float3 samplePosFromCamera) {
        float3 surfAtmPos = atmosphereParams.camera_position + float3(0.0, atmosphereParams.bottom_radius, 0.0) + samplePosFromCamera;
        float viewHeight = max(length(surfAtmPos), atmosphereParams.bottom_radius + 0.001);
        float3 upVector = surfAtmPos / viewHeight;
        float sunZenithCosAngle = dot(upVector, atmosphereParams.sun_direction);

        float2 uv;
        transmittanceParamsToUv(atmosphereParams, viewHeight, sunZenithCosAngle, uv);
        return transmittanceLUT.SampleLevel(linearSampler, uv, 0).rgb;
    }
}

public struct CameraUniforms {
    public matrix<float, 3, 4> model;
    public float tan_half_fov;
    public float far;
    public float near;
    // Sub-pixel jitter in pixels for the current frame. `jitter_x` reuses
    // the slot formerly named `_padding`; `jitter_y` reuses the first slot
    // of `_padding2`. Layout must match the Rust `Uniforms` struct.
    public float jitter_x;
    // Previous-frame world-from-view matrix and FOV. Used by the primary RT
    // pass to project this frame's hit point through last frame's camera and
    // produce screen-space motion vectors for DLSS-RR.
    public matrix<float, 3, 4> prev_model;
    public float prev_tan_half_fov;
    public float jitter_y;
    float2 _padding2;

    public float2 jitterPixel(uint2 position) {
        float2 jitter = float2(jitter_x, jitter_y); 
        return float2(position) + 0.5 - jitter;
    }

    // Camera-space (view-local) primary ray direction for a given pixel.
    // View frame convention: -Z forward, +Y up, +X right.
    public float3 primaryRayDirCameraSpace(uint2 position, uint2 screenSize) {
        return rayDirCameraSpaceFromNDC(jitterPixel(position) / float2(screenSize), screenSize);
    }

    // Camera-space direction through a point of the image plane. `pixelNDC` is
    // in [0, 1]^2 with the origin at the top-left corner of the screen, the
    // same space `projectToPixel` returns (divided by `screenSize`).
    public float3 rayDirCameraSpaceFromNDC(float2 pixelNDC, uint2 screenSize) {
        float2 pixelCamera = 2 * pixelNDC - 1;
        pixelCamera.y *= -1;
        pixelCamera.x *= float(screenSize.x) / float(screenSize.y);
        pixelCamera *= tan_half_fov;
        return normalize(float3(pixelCamera, -1));
    }

    // World -> current view space (translation and rotation; -Z forward).
    public float3 worldToView(float3 worldPos) {
        matrix<float, 3, 3> rotation =
            float3x3(model[0].xyz, model[1].xyz, model[2].xyz);
        return mul(transpose(rotation), worldPos - cameraOrigin());
    }

    // World -> current view space, directions only.
    public float3 worldDirToView(float3 worldDir) {
        matrix<float, 3, 3> rotation =
            float3x3(model[0].xyz, model[1].xyz, model[2].xyz);
        return mul(transpose(rotation), worldDir);
    }

    // ── Screen UV / device Z <-> world, jitter-consistent ─────────────────
    //
    // The depth G-buffer at texel `p` was written by the primary ray through
    // the jittered image-plane point `(p + 0.5 - jitter) / screenSize`. These
    // helpers use a "screen UV" in which that texel occupies `[p, p + 1) /
    // screenSize` regardless of the jitter: a surface reconstructed from texel
    // `p` projects back into texel `p`, which the HZB trace relies on.
    //   screenUV = pixelNDC + jitter / screenSize
    //   deviceZ  = near / linearViewZ (reverse infinite Z, as the depth G-buffer)

    // Projects `worldPos` through the current camera. `inFront` is false when
    // the point is behind the camera, in which case the result is unusable.
    public float3 projectToScreenUVZ(float3 worldPos, uint2 screenSize, out bool inFront) {
        float3 viewPos = worldToView(worldPos);
        inFront = viewPos.z < -1e-6;
        float negZ = max(-viewPos.z, 1e-6);
        float aspect = float(screenSize.x) / float(screenSize.y);
        float2 ndc;
        ndc.x = (viewPos.x / negZ) / (tan_half_fov * aspect) * 0.5 + 0.5;
        ndc.y = -(viewPos.y / negZ) / tan_half_fov * 0.5 + 0.5;
        float2 uv = ndc + float2(jitter_x, jitter_y) / float2(screenSize);
        return float3(uv, near / negZ);
    }

    // Inverse of `projectToScreenUVZ`.
    public float3 worldPosFromScreenUVZ(float2 screenUV, float deviceZ, uint2 screenSize) {
        float2 pixelNDC = screenUV - float2(jitter_x, jitter_y) / float2(screenSize);
        float3 dirCam = rayDirCameraSpaceFromNDC(pixelNDC, screenSize);
        float linearViewZ = near / max(deviceZ, 1e-12);
        float t = linearViewZ / -dirCam.z;
        matrix<float, 3, 3> rotation =
            float3x3(model[0].xyz, model[1].xyz, model[2].xyz);
        return cameraOrigin() + mul(rotation, dirCam) * t;
    }

    // Device Z `worldPos` would have had in last frame's depth buffer.
    public float prevDeviceZ(float3 worldPos) {
        matrix<float, 3, 3> rotation =
            float3x3(prev_model[0].xyz, prev_model[1].xyz, prev_model[2].xyz);
        float3 origin = float3(prev_model[0].w, prev_model[1].w, prev_model[2].w);
        float3 viewPos = mul(transpose(rotation), worldPos - origin);
        return near / max(-viewPos.z, 1e-6);
    }

    // World-space primary ray direction for a given pixel.
    public float3 primaryRayDirWorldSpace(uint2 position, uint2 screenSize) {
        matrix<float, 3, 3> rotation =
            float3x3(model[0].xyz, model[1].xyz, model[2].xyz);
        return mul(rotation, primaryRayDirCameraSpace(position, screenSize));
    }
    // World-space camerea origin.
    // Translation column of the 3x4 world-from-view matrix.
    public float3 cameraOrigin() {
        return float3(model[0].w, model[1].w, model[2].w);
    }

    // Pixel coordinate of `worldPos` as seen through the previous frame's
    // camera. Used to compute screen-space motion vectors. Returns NaN-free
    // garbage if the point is behind the previous camera; callers should
    // ignore MVs in that case.
    public float2 prevPixelFromWorld(float3 worldPos, uint2 screenSize) {
        // model is world-from-view, orthonormal rotation in the upper 3x3 +
        // translation in the .w column. Inverse rotation is the transpose.
        matrix<float, 3, 3> rotation =
            float3x3(prev_model[0].xyz, prev_model[1].xyz, prev_model[2].xyz);
        float3 origin = float3(prev_model[0].w, prev_model[1].w, prev_model[2].w);
        float3 viewPos = mul(transpose(rotation), worldPos - origin);
        // -Z is forward, so a point in front of the camera has viewPos.z < 0.
        float negZ = max(-viewPos.z, 1e-6);
        float aspect = float(screenSize.x) / float(screenSize.y);
        float ndcX = (viewPos.x / negZ) / (prev_tan_half_fov * aspect) * 0.5 + 0.5;
        float ndcY = -(viewPos.y / negZ) / prev_tan_half_fov * 0.5 + 0.5;
        return float2(ndcX, ndcY) * float2(screenSize) - 0.5;
    }

    // Pixel coordinate of an infinitely-distant point along `worldDir` as seen
    // through the previous frame's camera. Same projection as
    // prevPixelFromWorld but ignores translation, so motion is purely from
    // camera rotation + FOV change. Used for sky and UI-occluded pixels where
    // there is no surface to reproject.
    public float2 prevPixelFromWorldDir(float3 worldDir, uint2 screenSize) {
        matrix<float, 3, 3> rotation =
            float3x3(prev_model[0].xyz, prev_model[1].xyz, prev_model[2].xyz);
        float3 viewDir = mul(transpose(rotation), worldDir);
        float negZ = max(-viewDir.z, 1e-6);
        float aspect = float(screenSize.x) / float(screenSize.y);
        float ndcX = (viewDir.x / negZ) / (prev_tan_half_fov * aspect) * 0.5 + 0.5;
        float ndcY = -(viewDir.y / negZ) / prev_tan_half_fov * 0.5 + 0.5;
        return float2(ndcX, ndcY) * float2(screenSize) - 0.5;
    }

    // Project a world-space point to a pixel coordinate through the CURRENT
    // frame's camera. Returns false when the point is behind the camera (the
    // caller should discard it). Mirrors prevPixelFromWorld but uses this
    // frame's model / FOV — used by the SHARC candidate-pool splat debug view.
    public bool projectToPixel(float3 worldPos, uint2 screenSize, out float2 pixel) {
        matrix<float, 3, 3> rotation =
            float3x3(model[0].xyz, model[1].xyz, model[2].xyz);
        float3 origin = float3(model[0].w, model[1].w, model[2].w);
        float3 viewPos = mul(transpose(rotation), worldPos - origin);
        // -Z is forward, so a point in front of the camera has viewPos.z < 0.
        if (viewPos.z >= -1e-6) {
            pixel = float2(0.0, 0.0);
            return false;
        }
        float negZ = -viewPos.z;
        float aspect = float(screenSize.x) / float(screenSize.y);
        float ndcX = (viewPos.x / negZ) / (tan_half_fov * aspect) * 0.5 + 0.5;
        float ndcY = -(viewPos.y / negZ) / tan_half_fov * 0.5 + 0.5;
        pixel = float2(ndcX, ndcY) * float2(screenSize);
        return true;
    }
}

// Per-TLAS-instance data uploaded by the PBR plugin's `update_prev_transforms`
// system. `previous_transform` is the entity's GlobalTransform from the
// previous frame (world-from-object). Used to reproject moving objects' hit
// points through last frame's camera for DLSS-RR motion vectors.
public struct PbrInstanceData {
    public matrix<float, 3, 4> previous_transform;
}

public struct PbrPipelineGbuffer {
// [Stages] lists every shader stage that touches each binding across the
    // engine's pipelines (the gbuffer is also read by a SHARC debug compute pass,
    // hence Compute on several). The codegen coalesces descriptor writes only
    // within a matching-stage run, so these must mirror the stages in
    // pbr.playout.ron / sharc.playout.ron.
    [Stages(STAGE_RAYGEN | STAGE_MISS | STAGE_CLOSEST_HIT)]
    public RWTexture2D<float4> albedoLinear;
[Stages(STAGE_RAYGEN | STAGE_MISS | STAGE_CLOSEST_HIT | STAGE_COMPUTE)]
    public Texture2D<float4> albedoSRGB;
    // Uint (R8G8B8A8_UINT) view aliasing `albedoLinear`'s storage. The .a (w)
    // channel is a per-pixel miss shader index: the primary pass records which
    // miss shader the shadow / final-gather passes feed into TraceRay's
    // MissShaderIndex (real hits store 0; sky/UI store 0xFF as a "no surface"
    // sentinel those passes skip). Must be uint32_t4, not uint8_t4 — Vulkan
    // storage images require a 32-bit sampled type even for an 8-bit format; an
    // 8-bit image sampled type is invalid SPIR-V and spirv-val rejects it.
    [Stages(STAGE_RAYGEN | STAGE_MISS | STAGE_CLOSEST_HIT)]
    public RWTexture2D<uint32_t4> albedoUint;
    [Stages(STAGE_RAYGEN | STAGE_MISS | STAGE_CLOSEST_HIT | STAGE_COMPUTE)]
    public RWTexture2D<float4> normalTexture;
    [Stages(STAGE_RAYGEN | STAGE_MISS | STAGE_CLOSEST_HIT | STAGE_COMPUTE)]
    public RWTexture2D<float> depthTexture;
    // The SDR target for Debug overlays and UI. Values are sRGB-encoded.
    [Stages(STAGE_RAYGEN | STAGE_MISS | STAGE_CLOSEST_HIT | STAGE_COMPUTE)]
    public RWTexture2D<float4> occlusionTexture;
    // Screen-space motion (in pixels): currentPixel - prevPixel. Consumed by
    // DLSS-RR; set to (0, 0) for sky/miss pixels.
    [Stages(STAGE_RAYGEN | STAGE_MISS | STAGE_CLOSEST_HIT | STAGE_COMPUTE)]
    public RWTexture2D<float2> motionVectorTexture;
    // Specular albedo (F0 reflectance) for the deferred specular lobe and
    // DLSS-RR's specular guide buffer. Written by the primary pass (metalness
    // split: dielectric F0 from IOR, lerped to albedo by metalness), read by the
    // shadow + final-gather passes. Perceptual roughness rides in
    // normalTexture.w (DLSS reads it there — RoughnessMode::Packed).
    [Stages(STAGE_RAYGEN | STAGE_MISS | STAGE_CLOSEST_HIT)]
    public RWTexture2D<float4> specularAlbedo;

    // Draws an opaque pixel of linear `color` into `occlusionTexture`.
    public void writeOverlay(uint2 pixel, float3 color) {
        // The sRGB transfer function, since the texture is bound through its
        // UNORM view. Saturated first, as the UNORM store would clamp anyway.
        color = saturate(color);
        float3 encoded = select(color <= 0.0031308, color * 12.92, 1.055 * pow(color, 1.0 / 2.4) - 0.055);
        occlusionTexture[pixel] = float4(encoded, 1.0);
    }
}

// ── Screen-space (HZB) gather trace ─────────────────────────────────────
//
// Per-frame parameters of the screen trace / compaction / indirect gather
// passes (screen_trace.slang, final_gather.slang). Layout MUST match
// `ScreenTraceConstantsRaw` in crates/pbr/src/screen_trace.rs.
public struct ScreenTraceConstants {
    // Screen UV -> HZB UV: renderExtent / (2 * hzbExtent). Mip 0 of the HZB is
    // half the power-of-two-rounded render extent, so the render area covers
    // only the `[0, hzbUvFactor]` part of the HZB texture.
    public float2 hzbUvFactor;
    public float2 hzbInvUvFactor;
    // 1 / hzbExtent (mip 0).
    public float2 hzbBaseTexelSize;
    public float2 renderExtent;
    // Traversal budget of `traceHzb`.
    public float maxIterations;
    // Surface thickness as a fraction of the hit's view depth (Lumen's
    // RelativeDepthThickness)...
    public float relativeDepthThickness;
    // ...and the same for the history depth test at the reprojected hit.
    public float historyDepthThickness;
    // Floor on the thickness in world units: voxel edges are all hard steps,
    // so a thin surface never gets thinner than about half a voxel.
    public float minWorldThickness;
    // Linear steps past a hit that must all stay in front of the HZB for the
    // hit to count as certain (0 disables the search).
    public uint numThicknessSteps;
    // World units the hardware ray backs up from the last screen-visible point.
    public float pullbackBias;
    public uint flags;
    public uint debugMode;
    // Per-frame sampling seed shared by the screen trace and the gather ray,
    // so both passes trace the same direction.
    public uint frameIndex;
    public uint hzbNumMips;
    // Debug view controls (see `ScreenTraceDebugMode` in screen_trace.rs).
    public uint debugMip;
    public float debugScale;
    // Mip-0 extent of the HZB, for clamped texel fetches.
    public uint2 hzbExtent;
    // Stop marching once fewer lanes of the wave than this are still
    // tracing; the rest fall through to the hardware ray (Lumen's
    // MinimumOccupancy). 0 disables the check.
    public uint minimumOccupancy;
    uint _pad;
}

// Screen tracing is on: the gather dispatch is 1D over `compactedTraceTexels`,
// sized by `compactedTraceArgs` through an indirect trace-rays call. Off:
// the gather covers the screen and every surface pixel takes a hardware ray
// from t = 0.
public static const uint SCREEN_TRACE_FLAG_ENABLED = 1u << 0;
// Last frame's depth and color are complete (FrameHistory::valid).
public static const uint SCREEN_TRACE_FLAG_HISTORY_VALID = 1u << 1;
// Screen-only gather: no hardware rays follow, so the screen trace resolves
// every pixel itself. Uncertain hits are accepted, misses take the sky
// radiance along the ray, and rejected hits contribute nothing.
public static const uint SCREEN_TRACE_FLAG_SCREEN_ONLY = 1u << 2;

// ── Trace record (one uint per pixel in `traceHit`) ─────────────────────
//
// Port of Lumen's EncodeProbeRayDistance (LumenScreenProbeCommon.ush): the
// resume distance is stored as a float with the flags packed into its three
// lowest mantissa bits and its sign bit, so reading the distance back is a
// mask and an `asfloat`.
public static const uint TRACE_HIT_FLAG_HIT = 1u << 0;        // accepted screen hit, no hardware ray
public static const uint TRACE_HIT_FLAG_UNCERTAIN = 1u << 1;  // hit behind a thin feature, retraced
public static const uint TRACE_HIT_FLAG_REJECTED = 1u << 2;   // hit failed the history / edge tests
public static const uint TRACE_HIT_FLAG_REACHED_MAX = 1u << 31;
public static const uint TRACE_HIT_DISTANCE_MASK = 0x7FFFFFF8u;

public uint encodeTraceHit(float distance, bool hit, bool uncertain, bool rejected, bool reachedMax) {
    uint encoded = asuint(max(distance, 0.0)) & TRACE_HIT_DISTANCE_MASK;
    encoded |= hit ? TRACE_HIT_FLAG_HIT : 0u;
    encoded |= uncertain ? TRACE_HIT_FLAG_UNCERTAIN : 0u;
    encoded |= rejected ? TRACE_HIT_FLAG_REJECTED : 0u;
    encoded |= reachedMax ? TRACE_HIT_FLAG_REACHED_MAX : 0u;
    return encoded;
}

public struct TraceHit {
    public float distance;
    public bool hit;
    public bool uncertain;
    public bool rejected;
    public bool reachedMax;
}

public TraceHit decodeTraceHit(uint encoded) {
    TraceHit t;
    t.distance = asfloat(encoded & TRACE_HIT_DISTANCE_MASK);
    t.hit = (encoded & TRACE_HIT_FLAG_HIT) != 0;
    t.uncertain = (encoded & TRACE_HIT_FLAG_UNCERTAIN) != 0;
    t.rejected = (encoded & TRACE_HIT_FLAG_REJECTED) != 0;
    t.reachedMax = (encoded & TRACE_HIT_FLAG_REACHED_MAX) != 0;
    return t;
}

// Payload of the final-gather ray. The gather dispatch is 1D over the
// compacted trace list, so the miss and closest-hit shaders learn the pixel
// they light from the payload rather than DispatchRaysIndex().
public struct [raypayload] FinalGatherPayload {
    public uint2 pixel : read(closesthit, miss) : write(caller);
}

public struct PbrPipelineParams {
    [Stages(STAGE_RAYGEN)]
    public RaytracingAccelerationStructure sceneBVH;
    [Stages(STAGE_RAYGEN | STAGE_MISS | STAGE_CLOSEST_HIT | STAGE_COMPUTE)]
    public ConstantBuffer<CameraUniforms> uniforms;
    [Stages(STAGE_RAYGEN | STAGE_MISS | STAGE_CLOSEST_HIT | STAGE_COMPUTE)]
    public RWTexture2D outputTexture;
    public PbrPipelineGbuffer gbuffer;
    public SkyAtmosphereParams sky;
    // Per-TLAS-instance data, indexed by InstanceIndex(). Uploaded once per
    // frame after the TLAS instance buffer; each entry holds the previous
    // frame's world-from-object transform.
    [Stages(STAGE_CLOSEST_HIT)]
    public StructuredBuffer<PbrInstanceData> perInstanceData;
    public SharcParams sharc;

    // Screen-space gather trace (bindings 25..31). Read by the screen-trace,
    // compaction and debug compute passes and by the final-gather ray-gen.
    // The set is a push-descriptor set, which this device caps at 32
    // descriptors, so the trace fetches texels (no point sampler) and the
    // history color goes through `sky.linearSampler` (bilinear, clamp).
    // Closest-depth HZB of this frame's depth G-buffer (hzb_build.slang).
    [Stages(STAGE_RAYGEN | STAGE_COMPUTE)]
    public Texture2D<float> hzbTexture;
    // Last frame's denoised radiance, display resolution.
    [Stages(STAGE_RAYGEN | STAGE_COMPUTE)]
    public Texture2D<float4> prevSceneColor;
    // Last frame's depth G-buffer, render resolution.
    [Stages(STAGE_RAYGEN | STAGE_COMPUTE)]
    public Texture2D<float> prevDepthTexture;
    [Stages(STAGE_RAYGEN | STAGE_COMPUTE)]
    public ConstantBuffer<ScreenTraceConstants> screenTrace;
    // Packed trace record per pixel (`encodeTraceHit`).
    [Stages(STAGE_RAYGEN | STAGE_COMPUTE)]
    public RWTexture2D<uint> traceHit;
    // Pixels needing a hardware ray, packed `x | y << 16`, in dispatch order.
    [Stages(STAGE_RAYGEN | STAGE_COMPUTE)]
    public RWStructuredBuffer<uint> compactedTraceTexels;
    // `[count, 1, 1]`: a VkTraceRaysIndirectCommandKHR for the gather.
    [Stages(STAGE_RAYGEN | STAGE_COMPUTE)]
    public RWStructuredBuffer<uint> compactedTraceArgs;


    // Ray-length `t` from the camera origin to the surface stored at `position`.
    // The depth gbuffer holds reverse-infinite-Z hardware depth: `near /
    // linearViewZ`, where linearViewZ is the perpendicular distance from the
    // camera plane to the surface. Sky/UI pixels store 0, which decodes to
    // `t = +infinity`; callers must gate on `depth == 0` before using `t`.
    public float pixelDistanceFromCamera(uint2 position, uint2 screenSize) {
        float3 dirCam = uniforms.primaryRayDirCameraSpace(position, screenSize);
        float depthHw = gbuffer.depthTexture[position];
        // linearViewZ = near / depthHw; t = linearViewZ / cos(angle_from_forward).
        // dirCam is unit with -Z forward, so cos = -dirCam.z (> 0).
        return uniforms.near / (depthHw * -dirCam.z);
    }

    // Reconstructs the world-space surface position for a pixel from the gbuffer.
    // Undefined for sky/UI pixels (depth == 0); callers must gate.
    public float3 pixelPositionWorldSpace(uint2 position, uint2 screenSize) {
        float3 dir = uniforms.primaryRayDirWorldSpace(position, screenSize);
        return uniforms.cameraOrigin() + dir * pixelDistanceFromCamera(position, screenSize);
    }

    // Pixel coordinate of an object-space hit point as seen through the
    // previous frame's camera, accounting for the instance's own motion.
    // `objectHit` is the hit point in object-local coordinates;
    // `instanceIndex` is the TLAS instance index (i.e. SPIR-V InstanceIndex()).
    public float2 prevPixelFromObjectHit(float3 objectHit, uint instanceIndex, uint2 screenSize) {
        float3x4 prevObjToWorld = perInstanceData[instanceIndex].previous_transform;
        float3 prevWorldHit = mul(prevObjToWorld, float4(objectHit, 1.0));
        return uniforms.prevPixelFromWorld(prevWorldHit, screenSize);
    }
}

// Force the ParameterBlock to descriptor set 0. Without this, Slang reserves
// its own set for ParameterBlock contents as soon as any shader declares
// other explicit `[vk::binding(_, 0)]` globals — which would land `params` at
// set 1 and the SHARC working buffers at set 0, requiring a multi-set pipeline
// layout (only one set per layout can be a push-descriptor set, so that path
// fails validation). Pinning it here keeps a single push-descriptor layout.
[vk::binding(0, 0)] public ParameterBlock<PbrPipelineParams> params;

public struct [raypayload] SharcRayPayload {
    public uint cacheSlot : read(closesthit, miss) : write(caller);
}
