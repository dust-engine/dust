
[[vk::binding(1)]] uniform RWTexture2D outputTexture;

cbuffer PaletteBuffer
{
    // The palette, using float4 for R, G, B, A
    static const float4 ColorPalette[10] =
        {
            // 0: Deep Red
            float4(0.85, 0.15, 0.25, 1.0),
            // 1: Bright Orange
            float4(1.00, 0.60, 0.00, 1.0),
            // 2: Golden Yellow
            float4(0.95, 0.90, 0.40, 1.0),
            // 3: Vivid Green
            float4(0.20, 0.75, 0.20, 1.0),
            // 4: Cyan/Aqua
            float4(0.10, 0.85, 0.85, 1.0),
            // 5: Royal Blue
            float4(0.20, 0.20, 0.80, 1.0),
            // 6: Deep Violet
            float4(0.55, 0.20, 0.80, 1.0),
            // 7: Hot Pink
            float4(0.95, 0.25, 0.65, 1.0),
            // 8: Light Gray
            float4(0.70, 0.70, 0.70, 1.0),
            // 9: Dark Brown
            float4(0.35, 0.25, 0.15, 1.0)
        };
};

[shader("closesthit")]
void closestHitShader(
) {
    outputTexture[DispatchRaysIndex().xy] = ColorPalette[PrimitiveIndex() % 10];
}



[shader("intersection")]
void intersectionShader()
{
    ReportHit(12.0, 0, 0);
}
