use glam::{Vec3, Vec4};
use std::borrow::Cow;

pub struct Material {
    pub name: Cow<'static, str>,
    pub scale: f32,
    pub diffuse: image::DynamicImage,
}

#[repr(C)]
pub(crate) struct MaterialDeviceLayout {
    pub(crate) scale: f32,
    pub(crate) diffuse: u16,
    pub(crate) normal: u16,
    _reserved1: f32,
    _reserved2: f32,
}

pub struct ColoredMaterial {
    pub name: Cow<'static, str>,
    pub scale: f32,
    pub diffuse: image::DynamicImage,
    pub color_palette: [Vec3; 128],
}

#[repr(C)]
pub(crate) struct ColoredMaterialDeviceLayout {
    pub(crate) scale: f32,
    pub(crate) diffuse: u16,
    pub(crate) normal: u16,
    _reserved1: f32,
    _reserved2: f32,
    pub(crate) palette: [Vec4; 128],
}
