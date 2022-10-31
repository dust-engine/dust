use std::{sync::Arc, collections::BTreeMap};

use bevy_asset::{Assets, Handle, AssetServer};
use bevy_utils::{HashMap, HashSet};
pub use dustash::ray_tracing::sbt::HitGroupType;

use crate::{shader::{SpecializedShaderHandle, Shader}, Device};
use crate::geometry::Geometry;
use crate::material::Material;
use std::hash::{Hash, Hasher};

/// A renderer may contain multiple pipelines, each having their own hitgroups, raygen, callable and miss shaders.
/// Each pipeline can dispatch multiple ray types.
/// It is recommended to use separate `RenderPipeline` instances for separate ray-tracing passes. Combining all possible
/// shaders into a single `RenderPipeline` reduces opportunities for shader compiler optimizations.
/// 
/// Minimize max_recursion_depth. By default 
pub struct RayTracingPipelineLayout {
    raygen_shaders: Vec<SpecializedShaderHandle>,
    miss_shaders: Vec<SpecializedShaderHandle>,
    callable_shaders: Vec<SpecializedShaderHandle>,
    hitgroups: BTreeMap<u32, Vec<HitGroup>>,
}
pub struct RayTracingPipelineLayoutShaderRef(usize);
impl RayTracingPipelineLayout {
    pub fn new() -> Self {
        Self {
            raygen_shaders: Vec::new(),
            miss_shaders: BTreeMap::new(),
            callable_shaders: BTreeMap::new(),
            hitgroups: BTreeMap::new(),
        }
    }
    /// Performance tips:
    /// - The raygen shader should contain only one call to trace_rays(...), and it should terminates
    /// immeidately after trace_rays(...).
    /// - Each RenderPipeline should not contain more than one raygen shader.
    pub fn add_raygen_shader(&mut self, shader: SpecializedShaderHandle) -> RayTracingPipelineLayoutShaderRef {
        let i = self.raygen_shaders.len();
        self.raygen_shaders.push(shader);
        RayTracingPipelineLayoutShaderRef(i)
    }

    pub fn add_miss_shader(&mut self, shader: SpecializedShaderHandle) -> RayTracingPipelineLayoutShaderRef {
        let i = self.miss_shaders.len();
        self.miss_shaders.push(shader);
        RayTracingPipelineLayoutShaderRef(i)
    }

    /// The corresponding hitgroup shaders will be called with parameter of type `T` when the ray hits
    /// an entity with geometry `G` and material `M`.
    pub fn add_hitgroup<G: Geometry, M: Material>(
        &mut self,
        ray_type: u32,
        intersection: Option<SpecializedShaderHandle>,
        closest_hit: Option<SpecializedShaderHandle>,
        any_hit: Option<SpecializedShaderHandle>,
    ) {
        self.hitgroups.entry(ray_type).or_default().push(HitGroup {
            ty: HitGroupType::Procedural,
            intersection,
            closest_hit,
            any_hit
        });
    }
    

    /// Performance tips:
    /// - Use callable shaders for compute tasks with high SIMD divergence. Calls to callable shaders
    ///   potentially go through shader repacking, which can improve coherence but comes with its own overhead.
    ///  
    pub fn add_callable_shader(&mut self, id: u32, shader: SpecializedShaderHandle) -> RayTracingPipelineLayoutShaderRef {
        let i = self.callable_shaders.len();
        self.callable_shaders.push(shader);
        RayTracingPipelineLayoutShaderRef(i)
    }
}

pub struct HitGroup {
    pub ty: HitGroupType,
    pub intersection: Option<SpecializedShaderHandle>,
    pub any_hit: Option<SpecializedShaderHandle>,
    pub closest_hit: Option<SpecializedShaderHandle>,
}
impl HitGroup {
    /// Returns Some if all three hitgroup shaders are loaded. Otherwise return None.
    fn try_extract_shaders(
        &self,
        device: &Arc<dustash::Device>,
        shaders: &Assets<Shader>,
    ) -> Option<dustash::ray_tracing::sbt::HitGroup> {
        use dustash::shader::SpecializedShader as SpecializedShaderModule;

        let build_shader = |specialized_shader: &SpecializedShaderHandle| {
            let shader = shaders.get(&specialized_shader.shader)?;
            Some(SpecializedShaderModule {
                shader: Arc::new(shader.create(device.clone())),
                specialization: specialized_shader.specialization.clone(),
                entry_point: shader.entry_point.clone()
            })
        };
        let hit_group = dustash::ray_tracing::sbt::HitGroup {
            ty: self.ty,
            intersection_shader: if let Some(shader) = self.intersection.as_ref() {
                Some(build_shader(shader)?)
            } else {
                None
            },
            anyhit_shader: if let Some(shader) = self.any_hit.as_ref() {
                Some(build_shader(shader)?)
            } else {
                None
            },
            closest_hit_shader: if let Some(shader) = self.closest_hit.as_ref() {
                Some(build_shader(shader)?)
            } else {
                None
            },
        };
        Some(hit_group)
    }
}

pub struct PipelineCache {
    gpu_cache: dustash::pipeline::PipelineCache,
    pending_pipelines: HashMap<PipelineCacheEntry, ()>,
    built_pipelines: HashMap<PipelineCacheEntry, ()>,
}

#[derive(Hash, PartialEq, Eq)]
struct PipelineCacheEntry {
    pipeline_id: u64,
    // A sorted list of enabled (ray_type, hitgroup_index) pairs. For generalized pipeline, this is empty.
    enabled_hitgroups: Vec<(u32, usize)>,
}

impl PipelineCache {
    pub fn fetch(
        &self,
        shaders: Assets<Shader>,
        pipeline: &RayTracingPipeline,
        enabled_hitgroups: Vec<(u32, usize)>, // Assumed to be sorted
    ) {
        let hash_entry = PipelineCacheEntry {
            pipeline_id: pipeline.id,
            enabled_hitgroups,
        };
        // it can also be the case that even the shaders aren't completely loaded
        if let Some(cached_pipeline) = self.built_pipelines.get(&hash_entry) {
            return cached_pipeline;
        } else if enabled_hitgroups.len() > 0 {
            // Retry with a generic pipeline
            // TODO: queue specialized pipeline for build
            let generic_hash_entry = PipelineCacheEntry {
                pipeline_id: pipeline.id,
                enabled_hitgroups: Vec::new(),
            };
            if let Some(cached_pipeline) = self.built_pipelines.get(&hash_entry) {
                return cached_pipeline;
            }
        } else {
            // queue both generalized and specialized pipelines for build
            // print warning, generalized pipeline should be built beforehand
            // retur empty pipeline
        }
    }

    fn build_pipeline(
        &mut self,
        pipeline: &RayTracingPipeline,
        shaders: Assets<Shader>,
        device: &Device,
        enabled_hitgroups: Vec<(u32, usize)>, // Assumed to be sorted
        shader_module_cache: HashMap<Handle<Shader>, Arc<dustash::shader::Shader>>
    ) -> Option<()>{
        let hash_entry = PipelineCacheEntry {
            pipeline_id: pipeline.id,
            enabled_hitgroups,
        };
        if self.pending_pipelines.contains_key(&hash_entry) {
            return; // Pipeline is already being built
        }

        let materialize_shader = |handle: &SpecializedShaderHandle| {
            use bevy_utils::hashbrown::hash_map::Entry::*;
            let module = match shader_module_cache.entry(handle.clone_weak()) {
                Occupied(module) => module.get().clone(),
                Vacant(slot) => {
                    // If any of those shaders are unloaded, we can't build the pipeline
                    let shader: &Shader = shaders.get(&handle.shader)?;
                    let shader_module = shader.create((*device).clone());
                    let shader_module = Arc::new(shader_module);
                    slot.insert(shader_module.clone());
                    shader_module
                },
            };
            Some(dustash::shader::SpecializedShader {
                shader: module,
                specialization: handle.specialization.clone(),
                entry_point: shader.entry_point.clone()
            })
        };

        let raygen_shaders = pipeline.raygen_shaders.iter().map(materialize_shader).collect::<Option<Vec<_>>>()?.dedup();
        let miss_shaders = pipeline.miss_shaders.values().map(materialize_shader).collect::<Option<Vec<_>>>()?;
        let callable_shaders = pipeline.callable_shaders.values().map(materialize_shader).collect::<Option<Vec<_>>>()?;
        let hitgroups: Vec<dustash::ray_tracing::sbt::HitGroup> = pipeline.hitgroups.iter().map(|hitgroup| {
            hitgroup.try_extract_shaders();
        }).collect::<Option<Vec<_>>>()?;
        let sbt_layout = dustash::ray_tracing::sbt::SbtLayout::new(
            raygen_shaders.into_boxed_slice(),
            miss_shaders.into_boxed_slice(),
            callable_shaders.into_boxed_slice(),
            &hitgroups
        );
        todo!()
    }
}


pub trait RayTracingPipeline {
    fn max_recursion_depth(&self) -> u32 {
        1
    }
    fn raygen_shaders(&self, asset_server: &AssetServer) -> Vec<SpecializedShaderHandle>;
    fn miss_shaders(&self, asset_server: &AssetServer) -> Vec<SpecializedShaderHandle>;
    fn callable_shaders(&self, asset_server: &AssetServer) -> Vec<SpecializedShaderHandle>;
    fn build(&self, asset_server: &AssetServer) -> RayTracingPipelineBuildJob {
        RayTracingPipelineBuildJob {
            raygen_shader: self.raygen_shader(asset_server),
            miss_shaders: self.miss_shaders(asset_server),
            callable_shaders: self.callable_shaders(asset_server),
            max_recursion_depth: self.max_recursion_depth(),
        }
    }
}
trait Renderer {
    // fn new, initialize self with pipelines
    // fn update_sbts, update sbts for all pipelines. for each pipeline, for each entity, sbt.setHitgroup(), and assign offset to the entity.
    // but then, when setting hitgroups this also tie in with the tlas instances
}


// Testing stage:::

struct PrimaryRayPipeline {
    raygen: RayTracingPipelineLayoutShaderRef,
}
impl PrimaryRayPipeline {
    pub fn new(asset_server: AssetServer) -> Self {
        Self {
            raygen: SpecializedShaderHandle::new(asset_server.load("raygen.rgen"))
        }
    }
}
struct MyRenderer {
    primary_ray_pipeline: PrimaryRayPipeline,
}

impl MyRenderer {
    pub fn new(asset_server: AssetServer) {
        let primary_ray_pipeline = RayTracingPipeline::new(0);
        primary_ray_pipeline.add_raygen_shader(SpecializedShaderHandle::new(asset_server.load("raygen.rgen")));
    }
}

// Ok it's getting late. But basically, define traits for Pipeline and Renderer.
// Then, we can do pipeline.xxxx_shader to obtain a ShaderRef.
// When creating SBT, we can do SBT.set_callable(callable_index, shader_ref, data) so we get to reuse the same shader_ref but with different data.