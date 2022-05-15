use std::sync::Arc;

use bevy_asset::{AssetLoader, Handle, LoadedAsset};
use bevy_reflect::TypeUuid;
use dustash::{shader::SpecializationInfo, Device};

#[derive(TypeUuid)]
#[uuid = "ec052e5b-03ab-443f-9eac-b368526350fa"]
pub struct Shader {
    data: Box<[u32]>,
}

impl Shader {
    pub fn create(&self, device: Arc<Device>) -> dustash::shader::Shader {
        dustash::shader::Shader::from_spirv(device, &self.data)
    }
}

#[derive(Default)]
pub struct ShaderLoader;
impl AssetLoader for ShaderLoader {
    fn load<'a>(
        &'a self,
        bytes: &'a [u8],
        load_context: &'a mut bevy_asset::LoadContext,
    ) -> bevy_asset::BoxedFuture<'a, Result<(), anyhow::Error>> {
        assert!(bytes.len() % 4 == 0);
        let shader = Shader {
            data: unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const u32, bytes.len() / 4).into()
            },
        };

        Box::pin(async {
            load_context.set_default_asset(LoadedAsset::new(shader));
            Ok(())
        })
    }

    fn extensions(&self) -> &[&str] {
        &["spv"]
    }
}

#[derive(Clone)]
pub struct SpecializedShader {
    pub shader: Handle<Shader>,
    pub specialization: Option<SpecializationInfo>,
}
