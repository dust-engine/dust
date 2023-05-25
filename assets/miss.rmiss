
#version 460
#include "standard.glsl"

void main() {
    float ambient_light_intensity = 20.0;
    vec3 sky_blue_color = vec3(0.53, 0.80, 0.98);
    imageStore(u_illuminance, ivec2(gl_LaunchIDEXT.xy), vec4(vec3(ambient_light_intensity), 1.0));
    imageStore(u_albedo, ivec2(gl_LaunchIDEXT.xy), vec4(sky_blue_color, 1.0));
}
