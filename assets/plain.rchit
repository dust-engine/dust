#version 460
#extension GL_EXT_ray_tracing : require

struct RayPayload {
    vec3 color;
};

layout(location = 0) rayPayloadInEXT RayPayload primaryRayPayload;

void main() {
    primaryRayPayload.color = vec3(1.0, 0.0, 0.0);
}
