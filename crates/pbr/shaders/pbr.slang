uniform RaytracingAccelerationStructure sceneBVH;
uniform RWTexture2D outputTexture;
ConstantBuffer<Uniforms> uniforms;

struct Uniforms {
    matrix<float, 3, 4> model;
    float tan_half_fov;
    float far;
    float near;
    float _padding;
}

struct [raypayload] RayPayload {
    float color : read(caller) : write(caller, closesthit, miss);
}

float3 cameraDir(uint2 position, uint2 screenSize, float tan_half_fov, matrix<float, 3, 3> rotation) {
    float2 pixelNDC = (float2(position) + 0.5) / float2(screenSize);
    float2 pixelCamera = 2 * pixelNDC - 1;
    pixelCamera.y *= -1;
    pixelCamera.x *= float(screenSize.x) / float(screenSize.y);
    pixelCamera *= tan_half_fov;

    return mul(rotation, float3(pixelCamera, -1));
}

[shader("raygeneration")]
void rayGenShader() {
    // outputTexture currently stores occluding shapes.
    float4 originalColor = outputTexture[DispatchRaysIndex().xy];
    if (originalColor.a == 1.0) {
        // Solid color. Skip.
        return;
    }
    // Transparent color. Proceed.
    RayDesc ray;
    ray.Direction = cameraDir(
        DispatchRaysIndex().xy,
        DispatchRaysDimensions().xy,
        uniforms.tan_half_fov,
        float3x3(uniforms.model[0].xyz, uniforms.model[1].xyz, uniforms.model[2].xyz)
    );
    ray.Origin = float3(uniforms.model[0].w, uniforms.model[1].w, uniforms.model[2].w);
    ray.TMin = uniforms.near;
    ray.TMax = uniforms.far;

    RayPayload payload;
    payload.color = 0.0;

    TraceRay(sceneBVH, RAY_FLAG_NONE, ~0, 0, 1, 0, ray, payload);
}


[shader("miss")]
void missShader() {
    outputTexture[DispatchRaysIndex().xy] = float4(0.5, 0.0, 0.0, 1.0);
}
