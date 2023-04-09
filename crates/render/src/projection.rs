use bevy_ecs::prelude::Component;

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
            far: 1000.0,
        }
    }
}
