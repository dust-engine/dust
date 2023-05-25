
#version 460
#include "standard.glsl"


layout(location = 0) rayPayloadInEXT struct Payload {
    vec3 normal;
} payload;

void main() {
    float strength = 120.0; // FIXME: the light strength.

    vec3 illuminance = vec3(strength) * dot(payload.normal, gl_WorldRayDirectionEXT);
    imageStore(u_illuminance, ivec2(gl_LaunchIDEXT.xy), vec4(illuminance, 1.0));
}
