#include "./sky_atmosphere/atmosphere_common.slang"

struct SkyAtmosphereParams {
    ConstantBuffer<AtmosphereParams> atmosphereParams;
    Texture2D<float4> transmittanceLUT;
    Texture2D<float4> skyViewLUT;
    SamplerState linearSampler;
}


struct Uniforms {
    matrix<float, 3, 4> model;
    float tan_half_fov;
    float far;
    float near;
    float _padding;
}


struct PbrPipelineGbuffer {
    RWTexture2D<float4> albedoLinear;
    Texture2D<float4> albedoSRGB;
    RWTexture2D<float4> normalTexture;
    RWTexture2D<float> depthTexture;
    RWTexture2D<float4> occlusionTexture;
}

struct PbrPipelineParams {
    RaytracingAccelerationStructure sceneBVH;
    ConstantBuffer<Uniforms> uniforms;
    RWTexture2D outputTexture;
    PbrPipelineGbuffer gbuffer;
    SkyAtmosphereParams sky;
}
