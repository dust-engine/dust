use std::{collections::HashMap, path::PathBuf};

use bevy_app::Plugin;
use bevy_asset::{
    saver::{AssetSaver, SavedAsset},
    Asset, AssetLoader, AssetPath, LoadContext, LoadDirectError,
};
use bevy_reflect::TypePath;
use bevy_utils::thiserror::Error;
use futures_lite::{AsyncReadExt, AsyncWriteExt};
use shaderc::ResolvedInclude;

use super::SpirvLoader;

#[derive(TypePath, Asset)]
pub struct GlslShaderSource {
    pub source: String,
}

#[derive(TypePath, Asset)]
pub struct SpirvShaderSource {
    source: Vec<u8>,
}

/// Asset loader that loads GLSL source code as is.
pub struct GlslSourceLoader;

impl AssetLoader for GlslSourceLoader {
    type Asset = GlslShaderSource;
    type Settings = ();
    type Error = std::io::Error;
    fn load<'a>(
        &'a self,
        reader: &'a mut bevy_asset::io::Reader,
        _settings: &'a Self::Settings,
        _load_context: &'a mut bevy_asset::LoadContext,
    ) -> bevy_utils::BoxedFuture<'a, Result<Self::Asset, Self::Error>> {
        Box::pin(async move {
            let mut source = String::new();
            reader.read_to_string(&mut source).await?;

            Ok(GlslShaderSource { source })
        })
    }

    fn extensions(&self) -> &[&str] {
        &["glsl"]
    }
}

#[derive(Debug, Error)]
pub enum ShadercLoadingError {
    #[error("io error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("shaderc error: {0}")]
    ShadercError(#[from] shaderc::Error),
    #[error("shaderc error: {0}")]
    LoadingError(#[from] LoadDirectError),
}

/// Asset loader that compiles the GLSL source code into SPIR-V using Shaderc.
pub struct GlslShadercCompiler;
impl AssetLoader for GlslShadercCompiler {
    type Asset = SpirvShaderSource;

    type Settings = ();
    type Error = ShadercLoadingError;

    fn extensions(&self) -> &[&str] {
        &[
            "rgen", "rmiss", "rchit", "rahit", "rint", "frag", "vert", "comp",
        ]
    }

    fn load<'a>(
        &'a self,
        reader: &'a mut bevy_asset::io::Reader,
        _settings: &'a Self::Settings,
        ctx: &'a mut LoadContext,
    ) -> bevy_utils::BoxedFuture<'a, Result<Self::Asset, Self::Error>> {
        use shaderc::ShaderKind;
        let kind = if let Some(ext) = ctx.asset_path().get_full_extension() {
            match ext.as_str() {
                "rgen" => ShaderKind::RayGeneration,
                "rahit" => ShaderKind::AnyHit,
                "rchit" => ShaderKind::ClosestHit,
                "rint" => ShaderKind::Intersection,
                "rmiss" => ShaderKind::Miss,
                "comp" => ShaderKind::Compute,
                "vert" => ShaderKind::Vertex,
                "frag" => ShaderKind::Fragment,
                _ => ShaderKind::InferFromSource,
            }
        } else {
            ShaderKind::InferFromSource
        };

        Box::pin(async move {
            let mut includes = HashMap::new();
            let source = {
                let mut s = String::new();
                reader.read_to_string(&mut s).await?;
                s
            };

            let mut pending_sources = vec![("".to_string(), source.clone())];

            while !pending_sources.is_empty() {
                let (filename, source) = pending_sources.pop().unwrap();
                for (included_filename, ty) in source.lines().filter_map(match_include_line) {
                    if includes.contains_key(included_filename) {
                        continue;
                    }
                    let path = match ty {
                        shaderc::IncludeType::Relative => ctx
                            .path()
                            .parent()
                            .unwrap()
                            .join(&filename)
                            .join(included_filename),
                        shaderc::IncludeType::Standard => included_filename.into(),
                    };
                    let normalized_path = normalize_path(&path);
                    let inc = ctx
                        .load_direct(AssetPath::from_path(&normalized_path))
                        .await?;
                    let source: &GlslShaderSource = inc.get().unwrap();
                    pending_sources.push((included_filename.to_string(), source.source.clone()));
                }
                if !filename.is_empty() {
                    includes.insert(filename, source);
                }
            }

            use shaderc::{CompileOptions, Compiler};
            let compiler = Compiler::new().unwrap();

            let binary = {
                let mut options = CompileOptions::new().unwrap();
                options.set_target_spirv(shaderc::SpirvVersion::V1_6);
                options.set_target_env(
                    shaderc::TargetEnv::Vulkan,
                    rhyolite::Version::new(0, 1, 3, 0).as_raw(),
                );
                options.set_include_callback(|source_name, _ty, _, _include_depth| {
                    Ok(ResolvedInclude {
                        resolved_name: source_name.to_string(),
                        content: includes.get(source_name).ok_or("file not found")?.clone(),
                    })
                });
                options.set_source_language(shaderc::SourceLanguage::GLSL);
                options.set_forced_version_profile(460, shaderc::GlslProfile::Core);
                options.set_optimization_level(shaderc::OptimizationLevel::Performance);
                options.set_generate_debug_info();
                let binary_result =
                    compiler.compile_into_spirv(&source, kind, "", "main", Some(&options))?;
                binary_result.as_binary_u8().to_vec()
            };
            Ok(SpirvShaderSource { source: binary })
        })
    }
}

struct SpirvSaver;
impl AssetSaver for SpirvSaver {
    type Asset = SpirvShaderSource;
    type Settings = ();
    type OutputLoader = SpirvLoader;
    type Error = std::io::Error;
    fn save<'a>(
        &'a self,
        writer: &'a mut bevy_asset::io::Writer,
        asset: SavedAsset<'a, Self::Asset>,
        _settings: &'a Self::Settings,
    ) -> bevy_utils::BoxedFuture<Result<(), std::io::Error>> {
        Box::pin(async move {
            writer.write_all(&asset.source).await?;
            Ok(())
        })
    }
}

struct GlslSaver;
impl AssetSaver for GlslSaver {
    type Asset = GlslShaderSource;
    type Settings = ();
    type OutputLoader = GlslSourceLoader;
    type Error = std::io::Error;
    fn save<'a>(
        &'a self,
        writer: &'a mut bevy_asset::io::Writer,
        asset: SavedAsset<'a, Self::Asset>,
        _settings: &'a Self::Settings,
    ) -> bevy_utils::BoxedFuture<Result<(), std::io::Error>> {
        Box::pin(async move {
            writer.write_all(asset.source.as_bytes()).await?;
            Ok(())
        })
    }
}

pub struct GlslPlugin;
impl Plugin for GlslPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        use bevy_asset::AssetApp;
        app.init_asset::<GlslShaderSource>()
            .init_asset::<SpirvShaderSource>()
            .register_asset_loader(GlslSourceLoader)
            .register_asset_loader(PlayoutGlslLoader)
            .register_asset_loader(GlslShadercCompiler);
        if let Some(processor) = app
            .world
            .get_resource::<bevy_asset::processor::AssetProcessor>()
        {
            use bevy_asset::processor::LoadAndSave;

            // Load GLSL source, compile with shaderc, and save as SPIR-V
            type GlslToSpirv = LoadAndSave<GlslShadercCompiler, SpirvSaver>;
            processor.register_processor::<GlslToSpirv>(SpirvSaver.into());
            for ext in GlslShadercCompiler.extensions() {
                if *ext != "glsl" {
                    processor.set_default_processor::<GlslToSpirv>(ext);
                }
            }

            // Load Playout source, compile, and save as GLSL
            type PlayoutToGlsl = LoadAndSave<PlayoutGlslLoader, GlslSaver>;
            processor.register_processor::<PlayoutToGlsl>(GlslSaver.into());
            processor.set_default_processor::<PlayoutToGlsl>("playout");
        }
    }
}

fn match_include_line(line: &str) -> Option<(&str, shaderc::IncludeType)> {
    const PRAGMA_TOKEN: &'static str = "pragma";
    const INCLUDE_TOKEN: &'static str = "include";
    let mut s = line.trim();
    if !s.starts_with('#') {
        return None;
    }
    s = &s[1..];
    s = s.trim_start();

    if s.starts_with(PRAGMA_TOKEN) {
        s = &s[PRAGMA_TOKEN.len()..];
        s = s.trim_start();
    }
    if !s.starts_with(INCLUDE_TOKEN) {
        return None;
    }

    s = &s[INCLUDE_TOKEN.len()..];
    s = s.trim_start();

    let Some(first_char) = s.chars().nth(0) else {
        return None;
    };
    let ty = match first_char {
        '<' => shaderc::IncludeType::Standard,
        '"' => shaderc::IncludeType::Relative,
        _ => return None,
    };
    let last_char = match ty {
        shaderc::IncludeType::Relative => '"',
        shaderc::IncludeType::Standard => '>',
    };
    let Some(file_name) = s.split(last_char).skip(1).next() else {
        return None;
    };
    Some((file_name, ty))
}

pub fn normalize_path(path: &PathBuf) -> PathBuf {
    use std::path::Component;
    let mut components = path.components().peekable();
    let mut ret = if let Some(c @ Component::Prefix(..)) = components.peek().cloned() {
        components.next();
        PathBuf::from(c.as_os_str())
    } else {
        PathBuf::new()
    };

    for component in components {
        match component {
            Component::Prefix(..) => unreachable!(),
            Component::RootDir => {
                ret.push(component.as_os_str());
            }
            Component::CurDir => {}
            Component::ParentDir => {
                ret.pop();
            }
            Component::Normal(c) => {
                ret.push(c);
            }
        }
    }
    ret
}

#[derive(Error, Debug)]
pub enum PlayoutLoadingError {
    #[error("playout error: {0}")]
    PlayoutError(#[from] playout::Error),
    #[error("io error: {0}")]
    IoError(#[from] std::io::Error),
}

#[derive(Default)]
pub struct PlayoutGlslLoader;

impl AssetLoader for PlayoutGlslLoader {
    type Asset = GlslShaderSource;
    type Settings = ();
    type Error = PlayoutLoadingError;
    fn load<'a>(
        &'a self,
        reader: &'a mut bevy_asset::io::Reader,
        _settings: &'a Self::Settings,
        _load_context: &'a mut bevy_asset::LoadContext,
    ) -> bevy_utils::BoxedFuture<'a, Result<Self::Asset, Self::Error>> {
        Box::pin(async move {
            let mut source = String::new();
            reader.read_to_string(&mut source).await?;
            let module = playout::PlayoutModule::try_from(source.as_str())?;

            let mut source = String::new();
            module.show(&mut source);
            Ok(GlslShaderSource { source })
        })
    }

    fn extensions(&self) -> &[&str] {
        &["playout"]
    }
}
