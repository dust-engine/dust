layout(set = 0, binding = 0, rgba16f) uniform image2D u_illuminance;
layout(set = 0, binding = 1, rgba16f) uniform image2D u_illuminance_denoised;
layout(set = 0, binding = 2, rgb10_a2) uniform image2D u_albedo;
layout(set = 0, binding = 3, rgb10_a2) uniform image2D u_normal;
layout(set = 0, binding = 4, r32f) uniform image2D u_depth;
layout(set = 0, binding = 5, rgba16f) uniform image2D u_motion;
layout(set = 0, binding = 6, r32ui) uniform uimage2D u_voxel_id;
layout(set = 0, binding = 7) uniform texture2D blue_noise[6]; // [1d, 2d, unit2d, 3d, unit3d, unit3d_cosine]

layout(set = 0, binding = 9, std430) uniform CameraSettingsLastFrame {
    mat4 view_proj;
    mat4 inverse_view_proj;
    vec3 camera_view_col0;
    float position_x;
    vec3 camera_view_col1;
    float position_y;
    vec3 camera_view_col2;
    float position_z;
    float tan_half_fov;
    float far;
    float near;
    float padding;
} u_camera_last_frame;
layout(set = 0, binding = 10, std430) uniform CameraSettings {
    mat4 view_proj;
    mat4 inverse_view_proj;
    vec3 camera_view_col0;
    float position_x;
    vec3 camera_view_col1;
    float position_y;
    vec3 camera_view_col2;
    float position_z;
    float tan_half_fov;
    float far;
    float near;
    float padding;
} u_camera;
layout(set = 0, binding = 11, std430) buffer InstanceData {
    mat4 last_frame_transforms[];
} s_instances;

layout(set = 0, binding = 14) uniform accelerationStructureEXT accelerationStructure;
layout(push_constant) uniform PushConstants {
    // Indexed by block id
    uint rand;
    uint frameIndex;
} pushConstants;

