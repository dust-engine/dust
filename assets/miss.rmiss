
#version 460
#include "standard.glsl"

void main() {
    vec3 dir = normalize(gl_WorldRayDirectionEXT);
    vec3 sky_color_xyz = arhosek_sky_radiance(dir) + arhosek_sun_radiance(dir);

    imageStore(u_illuminance, ivec2(gl_LaunchIDEXT.xy), vec4(sky_color_xyz, 1.0));
    imageStore(u_albedo, ivec2(gl_LaunchIDEXT.xy), vec4(1.0));
    imageStore(u_depth, ivec2(gl_LaunchIDEXT.xy), vec4(0.0));

    imageStore(u_motion, ivec2(gl_LaunchIDEXT.xy), vec4(0.0, 0.0, 0.0, 0.0));
}
