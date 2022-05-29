#version 460
#extension GL_EXT_ray_tracing : require

struct RayPayload {
    vec3 color;
};

vec3 randomColorList[5] = {
    vec3(0.976, 0.906, 0.906),
    vec3(0.871, 0.839, 0.839),
    vec3(0.824, 0.796, 0.796),
    vec3(0.678, 0.627, 0.651),
    vec3(0.49, 0.576, 0.541),
};

layout(location = 0) rayPayloadInEXT RayPayload primaryRayPayload;

void main() {
    primaryRayPayload.color = randomColorList[gl_PrimitiveID % 5];
}
