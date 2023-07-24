use std::{path::Path, collections::HashMap};

use bevy_app::Plugin;
use bevy_reflect::TypePath;
use bevy_asset::{Asset, saver::AssetSaver, AssetLoader, ReadAssetBytesError, Handle, processor::ProcessContext};
use futures_lite::{AsyncReadExt, AsyncWriteExt};
use serde::{Deserialize, Serialize};
use shaderc::ResolvedInclude;
use crate::shader;

use super::SpirvLoader;


#[derive(TypePath, Asset)]
pub struct GlslShaderSource {
    source: String,
    kind: ShaderKind,
}


pub struct GlslLoader;

pub struct GlslCompiler;

#[derive(Clone, Copy)]
pub enum ShaderKind {
    Vertex,
    Fragment,
    Compute,
    Geometry,
    TessControl,
    TessEvaluation,

    /// Deduce the shader kind from `#pragma` directives in the source code.
    ///
    /// Compiler will emit error if `#pragma` annotation is not found.
    InferFromSource,

    RayGeneration,
    AnyHit,
    ClosestHit,
    Miss,
    Intersection,
    Callable,

    Task,
    Mesh,
}

impl AssetLoader for GlslLoader {
    type Asset = GlslShaderSource;
    type Settings = ();
    fn load<'a>(
            &'a self,
            reader: &'a mut bevy_asset::io::Reader,
            settings: &'a Self::Settings,
            load_context: &'a mut bevy_asset::LoadContext,
        ) -> bevy_asset::BoxedFuture<'a, Result<Self::Asset, bevy_asset::Error>> {
            let kind = if let Some(ext) = load_context.asset_path().get_full_extension() {
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
            let mut source = String::new();
            reader.read_to_string(&mut source).await?;

            Ok(GlslShaderSource {
                source,
                kind,
            })
        })
    }

    fn extensions(&self) -> &[&str] {
        &[
            "glsl",
            "rgen",
            "rmiss",
            "rchit",
            "rahit",
            "rint",
            "frag",
            "vert",
            "comp"
        ]
    }
}

impl AssetSaver for GlslCompiler {
    type Asset = GlslShaderSource;

    type Settings = ();

    type OutputLoader = SpirvLoader;

    fn save<'a>(
        &'a self,
        writer: &'a mut bevy_asset::io::Writer,
        asset: bevy_asset::saver::SavedAsset<'a, Self::Asset>,
        settings: &'a Self::Settings,
        ctx: &'a mut ProcessContext
    ) -> bevy_asset::BoxedFuture<'a, Result<Self::Settings, bevy_asset::Error>> {
        use shaderc::ShaderKind as SK;
        let kind = match asset.kind {
            ShaderKind::AnyHit => SK::AnyHit,
            ShaderKind::Vertex => SK::Vertex,
            ShaderKind::Fragment => SK::Fragment,
            ShaderKind::Compute =>SK::Compute,
            ShaderKind::Geometry => SK::Geometry,
            ShaderKind::TessControl => SK::TessControl,
            ShaderKind::TessEvaluation => SK::TessEvaluation,
            ShaderKind::InferFromSource => SK::InferFromSource,
            ShaderKind::RayGeneration => SK::RayGeneration,
            ShaderKind::ClosestHit => SK::ClosestHit,
            ShaderKind::Miss => SK::Miss,
            ShaderKind::Intersection => SK::Intersection,
            ShaderKind::Callable => SK::Callable,
            ShaderKind::Task =>SK::Task,
            ShaderKind::Mesh => SK::Mesh,
        };

        Box::pin(async move {
            let mut includes = HashMap::new();

            let mut pending_sources = vec![("".to_string(), asset.source.clone())];

            while !pending_sources.is_empty() {
                let (filename, source) = pending_sources.pop().unwrap();
                for (included_filename, ty) in source.lines().filter_map(match_include_line) {
                    if includes.contains_key(included_filename) {
                        continue;
                    }
                    let Ok(inc) = ctx.load_direct(included_filename).await else {
                        continue;
                    };
                    let Some(source): Option<&GlslShaderSource> = inc.get() else {
                        continue;
                    };
                    pending_sources.push((included_filename.to_string(), source.source.clone()));
                }
                if !filename.is_empty() {
                    includes.insert(filename, source);
                }
            }

            
            use shaderc::{
                Compiler,
                CompileOptions
            };
            let mut compiler = Compiler::new().unwrap();
    
            let binary = {
                let mut options = CompileOptions::new().unwrap();
                options.set_target_spirv(shaderc::SpirvVersion::V1_6);
                options.set_target_env(shaderc::TargetEnv::Vulkan, rhyolite::Version::new(0, 1, 3, 0).as_raw());
                options.set_include_callback(|source_name, ty, _, include_depth| {
                    Ok(ResolvedInclude {
                        resolved_name: source_name.to_string(),
                        content: includes.get(source_name).ok_or("file not found")?.clone(),
                    })
                });
                let binary_result = compiler.compile_into_spirv(
                    &asset.source, kind,
                    "", "main", Some(&options))?;
                    binary_result.as_binary_u8().to_vec()
            };
            writer.write_all(&binary).await?;
            Ok(())
        })
    }
}

pub struct GlslPlugin;
impl Plugin for GlslPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        use bevy_asset::AssetApp;
        app
        .init_asset::<GlslShaderSource>()
        .register_asset_loader(GlslLoader);
        if let Some(processor) = app.world.get_resource::<bevy_asset::processor::AssetProcessor>() {
            type P = bevy_asset::processor::LoadAndSave<GlslLoader, GlslCompiler>;
            processor.register_processor::<P>(
                GlslCompiler.into(),
            );
            for ext in GlslLoader.extensions() {
                if *ext != "glsl" {
                    processor.set_default_processor::<P>(ext);
                }
            };
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
        _ => return None
    };
    let last_char = match ty {
        shaderc::IncludeType::Relative => '"',
        shaderc::IncludeType::Standard => '>'
    };
    let Some(file_name) = s.split(last_char).skip(1).next() else {
        return None;
    };
    Some((file_name, ty))
}