#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

#include "shaders/utils/sky.glsl"

struct RayPayload {
    vec3 color;
    float hitT;
};

layout(location = 0) rayPayloadInEXT RayPayload primaryRayPayload;

const mat3 XYZ_2_RGB = (mat3(
     3.2404542,-0.9692660, 0.0556434,
    -1.5371385, 1.8760108,-0.2040259,
    -0.4985314, 0.0415560, 1.0572252
));

void main() {
    vec3 sky_color_xyz = arhosek_sky_radiance(normalize(gl_WorldRayDirectionEXT));
    primaryRayPayload.color = sky_color_xyz;
    primaryRayPayload.hitT = 0.0;
}
