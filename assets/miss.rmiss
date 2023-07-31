
#version 460
#include "standard.glsl"

void main() {
    vec3 dir = normalize(gl_WorldRayDirectionEXT);
    vec3 sky_color_xyz = arhosek_sky_radiance(dir) + arhosek_sun_radiance(dir);

    imageStore(u_illuminance, ivec2(gl_LaunchIDEXT.xy), vec4(sky_color_xyz, 1.0));
    imageStore(u_albedo, ivec2(gl_LaunchIDEXT.xy), vec4(1.0));
    imageStore(u_depth, ivec2(gl_LaunchIDEXT.xy), vec4(0.0));

    vec2 hitPointScreen = (vec2(gl_LaunchIDEXT.xy) + vec2(0.5)) / vec2(gl_LaunchSizeEXT.xy);
    vec2 hitPointNDC = hitPointScreen * 2.0 - 1.0;
    hitPointNDC.y *= -1.0;

    vec4 hitPointNDCLastFrame = u_camera_last_frame.view_proj * (u_camera.inverse_view_proj * vec4(hitPointNDC, 0.0, 1.0));
    vec3 hitPointNDCLastFrameNormalized = hitPointNDCLastFrame.xyz / hitPointNDCLastFrame.w;
    hitPointNDCLastFrameNormalized.y *= -1.0;
    vec2 hitPointScreenLastFrame = ((hitPointNDCLastFrameNormalized + 1.0) / 2.0).xy;

    vec2 motionVector = hitPointScreenLastFrame - hitPointScreen;
    imageStore(u_motion, ivec2(gl_LaunchIDEXT.xy), vec4(motionVector, 0.0, 0.0));



    Reservoir reservoir;
    reservoir.sample_count = 0;
    reservoir.total_weight = 0;
    reservoir.current_sample.voxel_id = 0xFFFFFFFF;
    s_reservoirs_prev.reservoirs[gl_LaunchIDEXT.x * gl_LaunchSizeEXT.y + gl_LaunchIDEXT.y] = reservoir;

}
