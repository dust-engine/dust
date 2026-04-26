import sky_atmosphere;
module pbr_common;

public struct SkyAtmosphereParams {
    public ConstantBuffer<AtmosphereParams> atmosphereParams;
    public Texture2D<float4> transmittanceLUT;
    public Texture2D<float4> skyViewLUT;
    public SamplerState linearSampler;
}

public struct Uniforms {
    public matrix<float, 3, 4> model;
    public float tan_half_fov;
    public float far;
    public float near;
    float _padding;
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
    public ConstantBuffer<Uniforms> uniforms;
    public RWTexture2D outputTexture;
    public PbrPipelineGbuffer gbuffer;
    public SkyAtmosphereParams sky;
    public Texture2DArray<float4> blueNoiseCosine;
}

public ParameterBlock<PbrPipelineParams> params;
