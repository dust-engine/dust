use glam::Mat4;

#[derive(Debug)]
pub struct CameraProjection {
    pub fov: f32,
    pub near: f32,
    pub far: f32,
}

impl CameraProjection {
    pub fn get_projection_matrix(&self, aspect_ratio: f32,) -> Mat4 {
        Mat4::perspective_rh(self.fov, aspect_ratio, self.near, self.far)
    }
}

impl Default for CameraProjection {
    fn default() -> Self {
        CameraProjection {
            fov: std::f32::consts::PI / 4.0,
            near: 0.1,
            far: 10.0,
        }
    }
}
