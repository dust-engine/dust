vec3 camera_origin() {
    return vec3(u_camera.position_x, u_camera.position_y, u_camera.position_z);
}
vec3 camera_ray_dir() {
    const vec2 pixelNDC = (vec2(gl_LaunchIDEXT.xy) + vec2(0.5)) / vec2(gl_LaunchSizeEXT.xy);

    vec2 pixelCamera = 2 * pixelNDC - 1;
    pixelCamera.y *= -1;
    pixelCamera.x *= float(gl_LaunchSizeEXT.x) / float(gl_LaunchSizeEXT.y);
    pixelCamera *= u_camera.tan_half_fov;

    const mat3 rotationMatrix = mat3(u_camera.camera_view_col0, u_camera.camera_view_col1, u_camera.camera_view_col2);

    const vec3 pixelCameraWorld = rotationMatrix * vec3(pixelCamera, -1);
    return pixelCameraWorld;
}
