use glam::Vec3;
use std::borrow::Cow;

pub struct Material {
    pub name: Cow<'static, str>,
    pub scale: f32,
    pub diffuse: image::DynamicImage,
}

pub struct ColoredMaterial {
    pub name: Cow<'static, str>,
    pub scale: f32,
    pub diffuse: Option<image::DynamicImage>,
    pub color_palette: [Vec3; 256],
}
