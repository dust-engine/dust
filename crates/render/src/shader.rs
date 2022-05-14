use std::sync::Arc;

use bevy_asset::{AssetLoader, Handle, LoadedAsset};
use bevy_reflect::TypeUuid;
use dustash::{shader::SpecializationInfo, Device};

#[derive(TypeUuid)]
#[uuid = "ec052e5b-03ab-443f-9eac-b368526350fa"]
pub enum Shader {
    Spirv(Box<[u32]>),
    Glsl(String),
}

impl Shader {
    pub fn create(&self, device: Arc<Device>) -> dustash::shader::Shader {
        match self {
            Shader::Spirv(data) => dustash::shader::Shader::from_spirv(device, &data),
            Shader::Glsl(data) => dustash::shader::Shader::from_glsl(device, data),
        }
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
        println!("Loaded shader");
        let is_spv = if let Some(ext) = load_context.path().extension() {
            ext == "spv"
        } else {
            let magic_number = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
            magic_number == 0x07230203
        };
        let shader = if is_spv {
            assert_eq!(bytes.len() % 4, 0);
            Shader::Spirv(unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const u32, bytes.len() / 4).into()
            })
        } else {
            Shader::Glsl(String::from_utf8(bytes.into()).unwrap())
        };

        Box::pin(async {
            load_context.set_default_asset(LoadedAsset::new(shader));
            Ok(())
        })
    }

    fn extensions(&self) -> &[&str] {
        &[
            "glsl",  // Generic GLSL code
            "vert",  // Tessellation control shader
            "tesc",  // Tessellation evaluation shader
            "geom",  // Geometry shader
            "frag",  // Fragment shader
            "comp",  // Compute shader
            "mesh",  // Mesh shader
            "task",  // Task shader
            "rgen",  // Ray generation shader
            "rint",  // Ray intersection shader
            "rahit", // Any hit shader
            "rchit", // Closest hit shader
            "rmiss", // Ray miss shader
            "rcall", // Ray callable shader
            "spv",   // Generic SPIR-V shader
        ]
    }
}

#[derive(Clone)]
pub struct SpecializedShader {
    pub shader: Handle<Shader>,
    pub specialization: Option<SpecializationInfo>,
}
