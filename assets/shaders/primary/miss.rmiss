#include "../headers/standard.glsl"
#include "../headers/layout.playout"
#include "../headers/color.glsl"
#include "../headers/sky.glsl"
#include "../headers/nrd.glsl"

void main() {
    vec3 dir = normalize(gl_WorldRayDirectionEXT);
    vec3 sky_color_xyz = arhosek_sky_radiance(dir) + arhosek_sun_radiance(dir);

    vec4 packed = REBLUR_FrontEnd_PackRadianceAndNormHitDist(sky_color_xyz, 100000.0);
    imageStore(img_illuminance_denoised, ivec2(gl_LaunchIDEXT.xy), packed);
    imageStore(img_albedo, ivec2(gl_LaunchIDEXT.xy), vec4(1.0));
    imageStore(img_depth, ivec2(gl_LaunchIDEXT.xy), vec4(1.0 / 0.0));
    imageStore(img_motion, ivec2(gl_LaunchIDEXT.xy), vec4(0.0));
}
