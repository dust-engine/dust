// The reference accumulator (the `Reference` upscaler backend): the running
// mean of every frame's radiance since the last reset, at display resolution.
// One thread per pixel.
//
// The mean is kept in 32-bit floats and updated incrementally,
// mean += (sample - mean) / n, rather than as a sum divided at the end, so it
// stays in the range of the radiance however many frames it averages.

struct AccumulateConstants {
    // Frames already averaged into `accumulation`. 0: its contents are
    // undefined, and this frame starts the mean.
    uint sampleCount;
    uint _pad;
    uint2 extent;
};

// This frame's radiance (`hdr_output`).
[vk::binding(0, 0)] [vk::image_format("rgba16f")] RWTexture2D<float4> radiance;
// The mean so far.
[vk::binding(1, 0)] [vk::image_format("rgba32f")] RWTexture2D<float4> accumulation;
// The mean, for the tone mapper (`hdr_denoised_output`).
[vk::binding(2, 0)] [vk::image_format("rgba16f")] RWTexture2D<float4> output;
[vk::binding(3, 0)] ConstantBuffer<AccumulateConstants> cb;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    uint2 pixel = id.xy;
    if (any(pixel >= cb.extent)) {
        return;
    }
    float3 sample = radiance[pixel].rgb;
    float3 mean = cb.sampleCount == 0 ? float3(0.0) : accumulation[pixel].rgb;
    mean += (sample - mean) / float(cb.sampleCount + 1);
    accumulation[pixel] = float4(mean, 1.0);
    output[pixel] = float4(mean, 1.0);
}
