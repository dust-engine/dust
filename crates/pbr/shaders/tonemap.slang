[[vk::binding(0, 0)]] RWTexture2D<float4> hdrInput;
[[vk::binding(1, 0)]] RWTexture2D<float4> ldrOutput;


// Simple ACES tone mapping
float3 acesToneMap(float3 color) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return saturate((color * (a * color + b)) / (color * (c * color + d) + e));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID: SV_DispatchThreadID,
          uniform float exposure) {
    uint2 coord = dispatchThreadID.xy;

    float4 hdr = hdrInput[coord];
    if (hdr.a > 0) {
        float3 color = acesToneMap(hdr.rgb * exposure);
        ldrOutput[coord] = float4(color, 1.0);
    }
}
