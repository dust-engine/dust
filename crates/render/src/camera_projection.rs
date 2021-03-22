use glam::Mat4;

pub struct CameraProjection {
    pub fov: f32,
    pub aspect_ratio: f32,
    pub near: f32,
    pub far: f32,
}

impl CameraProjection {
    pub fn get_projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov, self.aspect_ratio, self.near, self.far)
    }
    pub fn update_aspect_ratio(&mut self, width: f32, height: f32) {
        self.aspect_ratio = width / height;
    }
}

impl Default for CameraProjection {
    fn default() -> Self {
        CameraProjection {
            fov: std::f32::consts::PI / 4.0,
            near: 1.0,
            far: 1000.0,
            aspect_ratio: 1.0,
        }
    }
}
