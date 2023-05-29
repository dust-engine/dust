
#version 460
#include "standard.glsl"

void main() {
    float ambient_light_intensity = 20.0;
    vec3 sky_color_xyz = arhosek_sky_radiance(normalize(gl_WorldRayDirectionEXT));
    //vec3 sky_color_xyz = vec3(0.0);

    imageStore(u_illuminance, ivec2(gl_LaunchIDEXT.xy), vec4(sky_color_xyz, 1.0));
    imageStore(u_albedo, ivec2(gl_LaunchIDEXT.xy), vec4(1.0));
}
