import sky_atmosphere;
module pbr_common;
__exported import sharc;

public struct SkyAtmosphereParams {
    [Stages(STAGE_RAYGEN | STAGE_MISS)]
    public ConstantBuffer<AtmosphereParams> atmosphereParams;
    [Stages(STAGE_RAYGEN | STAGE_MISS)]
    public Texture2D<float4> transmittanceLUT;
    [Stages(STAGE_RAYGEN | STAGE_MISS)]
    public Texture2D<float4> skyViewLUT;
    [Stages(STAGE_RAYGEN | STAGE_MISS)]
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
        float2 pixelNDC = jitterPixel(position) / float2(screenSize);
        float2 pixelCamera = 2 * pixelNDC - 1;
        pixelCamera.y *= -1;
        pixelCamera.x *= float(screenSize.x) / float(screenSize.y);
        pixelCamera *= tan_half_fov;
        return normalize(float3(pixelCamera, -1));
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
    [Stages(STAGE_RAYGEN | STAGE_MISS | STAGE_CLOSEST_HIT)]
    public Texture2D<float4> albedoSRGB;
    [Stages(STAGE_RAYGEN | STAGE_MISS | STAGE_CLOSEST_HIT | STAGE_COMPUTE)]
    public RWTexture2D<float4> normalTexture;
    [Stages(STAGE_RAYGEN | STAGE_MISS | STAGE_CLOSEST_HIT | STAGE_COMPUTE)]
    public RWTexture2D<float> depthTexture;
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
}

public struct PbrPipelineParams {
    [Stages(STAGE_RAYGEN)]
    public RaytracingAccelerationStructure sceneBVH;
    [Stages(STAGE_RAYGEN | STAGE_MISS | STAGE_CLOSEST_HIT | STAGE_COMPUTE)]
    public ConstantBuffer<CameraUniforms> uniforms;
    [Stages(STAGE_RAYGEN | STAGE_MISS | STAGE_CLOSEST_HIT)]
    public RWTexture2D outputTexture;
    public PbrPipelineGbuffer gbuffer;
    public SkyAtmosphereParams sky;
    // Per-TLAS-instance data, indexed by InstanceIndex(). Uploaded once per
    // frame after the TLAS instance buffer; each entry holds the previous
    // frame's world-from-object transform.
    [Stages(STAGE_CLOSEST_HIT)]
    public StructuredBuffer<PbrInstanceData> perInstanceData;
    public SharcParams sharc;


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

// ── PBR shading helpers (metalness/roughness GGX) ─────────────────────────────
static const float kBrdfPi = 3.14159265358979;

// sRGB electro-optical transfer (gamma decode). The palette colors are authored
// in sRGB; a metal's F0 reflectance is its *linear* albedo, so the metal tint
// must be decoded before it enters the (linear) specular-albedo G-buffer.
public float3 srgbToLinear(float3 c) {
    float3 hi = pow(max(c + 0.055, 0.0) / 1.055, float3(2.4));
    float3 lo = c / 12.92;
    float3 useLo = step(c, float3(0.04045)); // 1 where c <= 0.04045
    return lerp(hi, lo, useLo);
}

// Trowbridge-Reitz (GGX) normal distribution. `a` is the GGX alpha (= perceptual
// roughness squared).
float D_GGX(float NoH, float a) {
    float a2 = a * a;
    float d = NoH * NoH * (a2 - 1.0) + 1.0;
    return a2 / max(kBrdfPi * d * d, 1e-7);
}

// Height-correlated Smith visibility — already folds in the 1/(4·NoV·NoL) of the
// Cook-Torrance denominator, so `specularGGX` returns f_spec directly.
float V_SmithGGXCorrelated(float NoV, float NoL, float a) {
    float a2 = a * a;
    float ggxV = NoL * sqrt(NoV * NoV * (1.0 - a2) + a2);
    float ggxL = NoV * sqrt(NoL * NoL * (1.0 - a2) + a2);
    return 0.5 / max(ggxV + ggxL, 1e-5);
}

float3 F_Schlick(float3 F0, float VoH) {
    float f = pow(saturate(1.0 - VoH), 5.0);
    return F0 + (1.0 - F0) * f;
}

// Cook-Torrance GGX specular BRDF value f_spec(V, L) for a surface frame N.
// `roughness` is perceptual (alpha = roughness²). The result includes the
// 1/(4·NoV·NoL) microfacet denominator via the visibility term.
public float3 specularGGX(float3 N, float3 V, float3 L, float roughness, float3 F0) {
    float3 H = normalize(V + L);
    float NoV = max(dot(N, V), 1e-4);
    float NoL = max(dot(N, L), 1e-4);
    float NoH = saturate(dot(N, H));
    float VoH = saturate(dot(V, H));
    float a = max(roughness * roughness, 1e-4);
    return D_GGX(NoH, a) * V_SmithGGXCorrelated(NoV, NoL, a) * F_Schlick(F0, VoH);
}

// Monte-Carlo specular weight for a radiance sample carried by a COSINE-sampled
// indirect ray (the final-gather ray). The estimator weight is f·cos/pdf with
// pdf = cos/π, so the cosine cancels and only π·f_spec remains — evaluated for
// the *primary* surface (N, V) and the gather direction wi. No extra ray: this
// reuses the existing diffuse gather sample.
public float3 cosineSpecularWeight(float3 N, float3 V, float3 wi, float roughness, float3 F0) {
    return kBrdfPi * specularGGX(N, V, wi, roughness, F0);
}

// Pre-integrated (environment) specular BRDF — the directional-hemispherical
// reflectance of the GGX specular lobe. This is the "specular albedo" DLSS-RR
// expects as its demodulation guide (NOT the raw F0): outSpecularAlbedo =
// EnvBRDFApprox2(F0, roughness², NoV). Analytic fit from Ray Tracing Gems
// ch. 32, as documented in the DLSS-RR integration guide.
public float3 EnvBRDFApprox2(float3 SpecularColor, float alpha, float NoV) {
    NoV = abs(NoV);
    float4 X;
    X.x = 1.0;
    X.y = NoV;
    X.z = NoV * NoV;
    X.w = NoV * X.z;
    float4 Y;
    Y.x = 1.0;
    Y.y = alpha;
    Y.z = alpha * alpha;
    Y.w = alpha * Y.z;
    float2x2 M1 = float2x2(0.99044, -1.28514, 1.29678, -0.755907);
    float3x3 M2 = float3x3(1.0, 2.92338, 59.4188, 20.3225, -27.0302, 222.592, 121.563, 626.13, 316.627);
    float2x2 M3 = float2x2(0.0365463, 3.32707, 9.0632, -9.04756);
    float3x3 M4 = float3x3(1.0, 3.59685, -1.36772, 9.04401, -16.3174, 9.22949, 5.56589, 19.7886, -20.2123);
    float bias = dot(mul(M1, X.xy), Y.xy) * rcp(dot(mul(M2, X.xyw), Y.xyw));
    float scale = dot(mul(M3, X.xy), Y.xy) * rcp(dot(mul(M4, X.xzw), Y.xyw));
    // Hack for specular reflectance of 0 (kills the bias for unlit/black F0).
    bias *= saturate(SpecularColor.g * 50.0);
    return SpecularColor * max(0.0, scale) + max(0.0, bias);
}
