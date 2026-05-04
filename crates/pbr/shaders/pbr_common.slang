import sky_atmosphere;
module pbr_common;

public struct SkyAtmosphereParams {
    public ConstantBuffer<AtmosphereParams> atmosphereParams;
    public Texture2D<float4> transmittanceLUT;
    public Texture2D<float4> skyViewLUT;
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

    // Camera-space (view-local) primary ray direction for a given pixel.
    // View frame convention: -Z forward, +Y up, +X right.
    public float3 primaryRayDirCameraSpace(uint2 position, uint2 screenSize) {
        float2 jitter = float2(jitter_x, jitter_y);
        float2 pixelNDC = (float2(position) + 0.5 - jitter) / float2(screenSize);
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
}

// Per-TLAS-instance data uploaded by the PBR plugin's `update_prev_transforms`
// system. `previous_transform` is the entity's GlobalTransform from the
// previous frame (world-from-object). Used to reproject moving objects' hit
// points through last frame's camera for DLSS-RR motion vectors.
public struct PbrInstanceData {
    public matrix<float, 3, 4> previous_transform;
}

public struct PbrPipelineGbuffer {
    public RWTexture2D<float4> albedoLinear;
    public Texture2D<float4> albedoSRGB;
    public RWTexture2D<float4> normalTexture;
    public RWTexture2D<float> depthTexture;
    public RWTexture2D<float4> occlusionTexture;
    // Screen-space motion (in pixels): currentPixel - prevPixel. Consumed by
    // DLSS-RR; set to (0, 0) for sky/miss pixels.
    public RWTexture2D<float2> motionVectorTexture;
}

public struct PbrPipelineParams {
    public RaytracingAccelerationStructure sceneBVH;
    public ConstantBuffer<CameraUniforms> uniforms;
    public RWTexture2D outputTexture;
    public PbrPipelineGbuffer gbuffer;
    public SkyAtmosphereParams sky;
    // Per-TLAS-instance data, indexed by InstanceIndex(). Uploaded once per
    // frame after the TLAS instance buffer; each entry holds the previous
    // frame's world-from-object transform.
    public StructuredBuffer<PbrInstanceData> perInstanceData;

    // Reconstructs the world-space surface position for a pixel from the gbuffer.
    public float3 pixelPositionWorldSpace(uint2 position, uint2 screenSize) {
        float3 dir = uniforms.primaryRayDirWorldSpace(position, screenSize);
        float3 origin = uniforms.cameraOrigin();
        float depth = gbuffer.depthTexture[position];
        float3 worldPos = origin + dir * depth;
        return worldPos;
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

public ParameterBlock<PbrPipelineParams> params;
