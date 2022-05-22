use bevy_ecs::{component::Component, system::{Query, Commands}, entity::Entity};
use bevy_transform::prelude::GlobalTransform;
use bevy_window::WindowId;
use glam::{Vec3, Mat3};

#[derive(Component)]
pub struct PerspectiveCamera {
    fov: f32,
    near: f32,
    far: f32,
    target: WindowId
}

#[repr(C)]
pub struct PerspectiveCameraParameters {
    pub camera_view_col0: [f32; 3],
    pub near: f32,
    pub camera_view_col1: [f32; 3],
    pub far: f32,
    pub camera_view_col2: [f32; 3],
    _padding: f32,

    pub camera_position: [f32; 3],
    pub tan_half_fov: f32,
}


#[derive(Component)]
pub struct ExtractedCamera {
    target: WindowId,
    params: PerspectiveCameraParameters
}

pub(crate) fn extract_camera_system(
    mut commands: Commands,
    query: Query<(Entity, &PerspectiveCamera, &GlobalTransform)>,
) {
    debug_assert_eq!(std::mem::size_of::<PerspectiveCameraParameters>(), std::mem::size_of::<f32>() * 16);
    for (entity, camera, transform) in query.iter() {
        let rotation_matrix = Mat3::from_quat(transform.rotation).to_cols_array_2d();
        let params = PerspectiveCameraParameters {
            camera_view_col0: rotation_matrix[0],
            near: camera.near,
            camera_view_col1: rotation_matrix[1],
            far: camera.far,
            camera_view_col2: rotation_matrix[2],
            _padding: 0.0,
            camera_position: transform.translation.to_array(),
            tan_half_fov: (camera.fov / 2.0).tan(),
        };
        commands.get_or_spawn(entity).insert(ExtractedCamera {
            target: camera.target,
            params,
        });
    }
}
