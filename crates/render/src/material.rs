use std::borrow::Cow;
use glam::Vec3;

pub struct Material {
    pub name: Cow<'static, str>,
    pub scale: f32,
    pub diffuse: Option<TextureRepoHandle>,
    pub normal: Option<TextureRepoHandle>
}

pub struct ColoredMaterial {
    pub name: Cow<'static, str>,
    pub scale: f32,
    pub diffuse: Option<TextureRepoHandle>,
    pub normal: Option<TextureRepoHandle>
    pub color_palette: [Vec3; 256]
}
