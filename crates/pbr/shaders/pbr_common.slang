import sky_atmosphere;
module pbr_common;

public struct SkyAtmosphereParams {
    public ConstantBuffer<AtmosphereParams> atmosphereParams;
    public Texture2D<float4> transmittanceLUT;
    public Texture2D<float4> skyViewLUT;
    public SamplerState linearSampler;
}

public struct CameraUniforms {
    public matrix<float, 3, 4> model;
    public float tan_half_fov;
    public float far;
    public float near;
    float _padding;

    // World-space primary ray direction for a given pixel.
    public float3 cameraDir(uint2 position, uint2 screenSize) {
        matrix<float, 3, 3> rotation =
            float3x3(model[0].xyz, model[1].xyz, model[2].xyz);
        float2 pixelNDC = (float2(position) + 0.5) / float2(screenSize);
        float2 pixelCamera = 2 * pixelNDC - 1;
        pixelCamera.y *= -1;
        pixelCamera.x *= float(screenSize.x) / float(screenSize.y);
        pixelCamera *= tan_half_fov;
        return normalize(mul(rotation, float3(pixelCamera, -1)));
    }
    // World-space camerea origin.
    // Translation column of the 3x4 world-from-view matrix.
    public float3 cameraOrigin() {
        return float3(model[0].w, model[1].w, model[2].w);
    }
}

public struct PbrPipelineGbuffer {
    public RWTexture2D<float4> albedoLinear;
    public Texture2D<float4> albedoSRGB;
    public RWTexture2D<float4> normalTexture;
    public RWTexture2D<float> depthTexture;
    public RWTexture2D<float4> occlusionTexture;
}

public struct PbrPipelineParams {
    public RaytracingAccelerationStructure sceneBVH;
    public ConstantBuffer<CameraUniforms> uniforms;
    public RWTexture2D outputTexture;
    public PbrPipelineGbuffer gbuffer;
    public SkyAtmosphereParams sky;
    public Texture2DArray<float4> blueNoiseCosine;

    // Reconstructs the world-space surface position for a pixel from the gbuffer.
    public float3 pixelPositionWorldSpace(uint2 position, uint2 screenSize) {
        float3 dir = uniforms.cameraDir(position, screenSize);
        float3 origin = uniforms.cameraOrigin();
        float depth = gbuffer.depthTexture[position];
        float3 worldPos = origin + dir * depth;
        return worldPos;
    }
}

public ParameterBlock<PbrPipelineParams> params;
