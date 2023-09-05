use std::{ffi::CStr, sync::Arc};

use bevy_asset::{Asset, AssetLoader, Handle};
use bevy_reflect::TypePath;
use futures_lite::AsyncReadExt;
use rhyolite::{ash::vk, cstr, shader::SpecializationInfo};

#[derive(TypePath, Asset)]
pub struct ShaderModule(Arc<rhyolite::shader::ShaderModule>);
impl ShaderModule {
    pub fn inner(&self) -> &Arc<rhyolite::shader::ShaderModule> {
        &self.0
    }
}

// TODO: Pipelines don't need to own the specialized shader once they've been created.
#[derive(Clone)]
pub struct SpecializedShader {
    pub stage: vk::ShaderStageFlags,
    pub flags: vk::PipelineShaderStageCreateFlags,
    pub shader: Handle<ShaderModule>,
    pub specialization_info: SpecializationInfo,
    pub entry_point: &'static CStr,
}
impl SpecializedShader {
    pub fn for_shader(shader: Handle<ShaderModule>, stage: vk::ShaderStageFlags) -> Self {
        Self {
            stage,
            flags: vk::PipelineShaderStageCreateFlags::empty(),
            shader,
            specialization_info: SpecializationInfo::default(),
            entry_point: cstr!("main"),
        }
    }
    pub fn with_const<T: Copy + 'static>(mut self, constant_id: u32, item: T) -> Self {
        self.specialization_info.push(constant_id, item);
        self
    }
}

pub struct SpirvLoader {
    device: rhyolite_bevy::Device,
}
impl SpirvLoader {
    pub(crate) fn new(device: rhyolite_bevy::Device) -> Self {
        Self { device }
    }
}
impl AssetLoader for SpirvLoader {
    type Asset = ShaderModule;
    type Settings = ();
    fn load<'a>(
        &'a self,
        reader: &'a mut bevy_asset::io::Reader,
        _settings: &'a Self::Settings,
        _load_context: &'a mut bevy_asset::LoadContext,
    ) -> bevy_asset::BoxedFuture<'a, Result<ShaderModule, bevy_asset::Error>> {
        let device = self.device.inner().clone();
        return Box::pin(async move {
            let mut bytes = Vec::new();
            reader.read_to_end(&mut bytes).await?;
            assert!(bytes.len() % 4 == 0);
            let bytes = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const u32, bytes.len() / 4)
            };
            let shader = rhyolite::shader::SpirvShader { data: bytes }.build(device)?;
            Ok(ShaderModule(Arc::new(shader)))
        });
    }

    fn extensions(&self) -> &[&str] {
        &["spv"]
    }
}
