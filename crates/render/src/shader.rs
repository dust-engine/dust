use std::{ffi::CStr, sync::Arc};

use bevy_asset::{AssetLoader, Handle, LoadedAsset};
use bevy_reflect::TypeUuid;
use rhyolite::{ash::vk, shader::SpecializationInfo};

#[derive(TypeUuid)]
#[uuid = "10c440f6-ca49-435b-998a-ee2c351044c4"]
pub struct ShaderModule(Arc<rhyolite::shader::ShaderModule>);
impl ShaderModule {
    pub fn inner(&self) -> &Arc<rhyolite::shader::ShaderModule> {
        &self.0
    }
}

pub struct SpecializedShader {
    pub stage: vk::ShaderStageFlags,
    pub flags: vk::PipelineShaderStageCreateFlags,
    pub shader: Handle<ShaderModule>,
    pub specialization_info: SpecializationInfo,
    pub entry_point: &'static CStr,
}

pub struct SpirvLoader {
    device: rhyolite_bevy::Device,
}
impl AssetLoader for SpirvLoader {
    fn load<'a>(
        &'a self,
        bytes: &'a [u8],
        load_context: &'a mut bevy_asset::LoadContext,
    ) -> bevy_asset::BoxedFuture<'a, Result<(), bevy_asset::Error>> {
        assert!(bytes.len() % 4 == 0);
        let bytes =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u32, bytes.len() / 4) };
        let device = self.device.inner().clone();
        return Box::pin(async move {
            let shader = rhyolite::shader::SpirvShader { data: bytes }.build(device)?;
            load_context.set_default_asset(LoadedAsset::new(ShaderModule(Arc::new(shader))));
            Ok(())
        });
    }

    fn extensions(&self) -> &[&str] {
        &["spv"]
    }
}
