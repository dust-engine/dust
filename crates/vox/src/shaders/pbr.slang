
[[vk::binding(1)]] uniform RWTexture2D outputTexture;

[shader("closesthit")]
void closestHitShader(
) {
    outputTexture[DispatchRaysIndex().xy] = float4(100.0, 0.0, 1.0, 1.0);
}



[shader("intersection")]
void intersectionShader()
{
    ReportHit(12.0, 0, 0);
}
