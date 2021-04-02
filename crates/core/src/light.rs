pub use glam::Vec3;

#[derive(Clone)]
pub struct SunLight {
    pub color: Vec3,
    pub(crate) _padding1: f32,
    pub dir: Vec3,
    pub(crate) _padding2: f32,
}

impl SunLight {
    pub fn new(color: Vec3, dir: Vec3) -> Self {
        SunLight {
            color,
            _padding1: 0.0,
            dir,
            _padding2: 0.0,
        }
    }
}
