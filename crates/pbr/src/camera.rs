use bevy::{a11y::accesskit::Vec2, ecs::{bundle::Bundle, component::Component}, math::{Mat4, UVec2, Vec3}, transform::{components::{GlobalTransform, Transform}, TransformBundle}};

#[derive(Clone, Component)]
pub struct PinholeProjection {
    pub fov: f32,

    /// The distance from the camera in world units of the viewing frustum's near plane.
    ///
    /// Objects closer to the camera than this value will not be visible.
    ///
    /// Defaults to a value of `0.1`.
    pub near: f32,

    /// The distance from the camera in world units of the viewing frustum's far plane.
    ///
    /// Objects farther from the camera than this value will not be visible.
    ///
    /// Defaults to a value of `1000.0`.
    pub far: f32,
}
impl Default for PinholeProjection {
    fn default() -> Self {
        Self {
            fov: std::f32::consts::FRAC_PI_4,
            near: 0.1,
            far: 10000.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct CameraUniform {
    pub view_proj: Mat4,
    pub inverse_view_proj: Mat4,
    pub camera_view_col0: Vec3,
    pub position_x: f32,
    pub camera_view_col1: Vec3,
    pub position_y: f32,
    pub camera_view_col2: Vec3,
    pub position_z: f32,
    pub tan_half_fov: f32,
    pub far: f32,
    pub near: f32,
    pub _padding: f32,
}
impl CameraUniform {
    pub fn from_transform_projection(
        transform: &GlobalTransform,
        projection: &PinholeProjection,
        aspect_ratio: f32,
    ) -> Self {
        let proj = {
            Mat4::perspective_infinite_reverse_rh(
                projection.fov,
                aspect_ratio,
                projection.near,
            )
        };
        let view_proj = proj * transform.compute_matrix().inverse();
        let view = &transform.affine().matrix3;
        CameraUniform {
            view_proj: view_proj,
            inverse_view_proj: view_proj.inverse(),
            camera_view_col0: view.x_axis.into(),
            camera_view_col1: view.y_axis.into(),
            camera_view_col2: view.z_axis.into(),
            near: projection.near,
            far: projection.far,
            _padding: 0.0,
            position_x: transform.translation().x,
            position_y: transform.translation().y,
            position_z: transform.translation().z,
            tan_half_fov: (projection.fov / 2.0).tan(),
        }
    }
}

#[derive(Bundle, Default)]
pub struct CameraBundle {
    pub projection: PinholeProjection,
    pub transform: TransformBundle,
}
