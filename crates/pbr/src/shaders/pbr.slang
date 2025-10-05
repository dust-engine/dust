
[shader("raygeneration")]
void rayGenShader(
    uniform RaytracingAccelerationStructure sceneBVH,
    uniform RWTexture2D outputTexture,
) {
    outputTexture[DispatchRaysIndex().xy] = float4(0.5, 100.0, 1.0, 1.0);
}


[shader("miss")]
void missShader(
    uniform RWTexture2D outputTexture,
) {
}
