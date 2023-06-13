use std::{
    collections::{BTreeMap, HashMap},
    ops::Deref,
    sync::Arc,
};

use bevy_asset::{Assets, Handle};
use bevy_tasks::AsyncComputeTaskPool;
use rhyolite::{
    ash::prelude::VkResult, HasDevice, PipelineLayout, RayTracingPipeline,
    RayTracingPipelineLibrary, RayTracingPipelineLibraryCreateInfo, RayTracingHitGroupType,
};

use crate::{
    deferred_task::{DeferredTaskPool, DeferredValue},
    material::Material,
    shader::{ShaderModule, SpecializedShader}, CachablePipeline, PipelineBuildInfo, CachedPipeline, PipelineCache,
};

use super::RayTracingPipelineCharacteristics;

struct RayTracingPipelineManagerMaterialInfo {
    instance_count: usize,
    pipeline_library: Option<DeferredValue<Arc<RayTracingPipelineLibrary>>>,
}
struct RayTracingPipelineManagerSpecializedPipelineDeferred {
    pipeline: CachedPipeline<rhyolite::RayTracingPipeline>,
    /// Mapping from (material_index, ray_type) to hitgroup index
    /// hitgroup index = hitgroup_mapping[material_index] + ray_type
    hitgroup_mapping: BTreeMap<u32, u32>,
}

#[derive(Clone, Copy)]
pub struct RayTracingPipelineManagerSpecializedPipeline<'a> {
    material_mapping: &'a HashMap<std::any::TypeId, usize>,
    pipeline: &'a Arc<rhyolite::RayTracingPipeline>,
    /// Mapping from (material_index, ray_type) to hitgroup index
    /// hitgroup index = hitgroup_mapping[material_index] + ray_type
    hitgroup_mapping: &'a BTreeMap<u32, u32>,

    /// A subset of all raytypes
    raytypes: &'a [u32],
}
impl<'a> HasDevice for RayTracingPipelineManagerSpecializedPipeline<'a> {
    fn device(&self) -> &Arc<rhyolite::Device> {
        self.pipeline.device()
    }
}

impl<'a> RayTracingPipelineManagerSpecializedPipeline<'a> {
    pub fn layout(&self) -> &Arc<PipelineLayout> {
        self.pipeline.layout()
    }
    pub fn pipeline(&self) -> &Arc<rhyolite::RayTracingPipeline> {
        self.pipeline
    }
    pub fn get_sbt_handle_for_material(
        &self,
        material_type: std::any::TypeId,
        raytype: u32,
    ) -> &[u8] {
        let material_index = *self.material_mapping.get(&material_type).unwrap() as u32;
        let local_raytype = self.raytypes.iter().position(|a| *a == raytype).unwrap();
        let hitgroup_index = self.hitgroup_mapping[&material_index] + local_raytype as u32;
        self.pipeline
            .sbt_handles()
            .hitgroup(hitgroup_index as usize)
    }
}

pub struct RayTracingPipelineManager {
    raytypes: Vec<u32>,
    /// A pipeline library containing raygen, raymiss, callable shaders
    pipeline_base_library: Option<DeferredValue<Arc<RayTracingPipelineLibrary>>>,
    pipeline_characteristics: Arc<RayTracingPipelineCharacteristics>,
    current_material_flag: u64,
    specialized_pipelines: BTreeMap<u64, RayTracingPipelineManagerSpecializedPipelineDeferred>,
    materials: Vec<RayTracingPipelineManagerMaterialInfo>,

    /// Raygen shaders, miss shaders, callable shaders
    shaders: Vec<SpecializedShader>,
}

impl CachablePipeline for RayTracingPipelineLibrary {
    type BuildInfo = RayTracingPipelineLibraryBuildInfo;
}

#[derive(Clone)]
pub struct RayTracingPipelineLibraryBuildInfo {
    pipeline_characteristics: Arc<RayTracingPipelineCharacteristics>,
    shaders: Vec<SpecializedShader>,
}
impl PipelineBuildInfo for RayTracingPipelineLibraryBuildInfo {
    type Pipeline = RayTracingPipelineLibrary;

    fn build(
        self,
        shader_store: &Assets<ShaderModule>,
        pipeline_cache: Option<&Arc<rhyolite::PipelineCache>>,
    ) -> DeferredValue<Arc<Self::Pipeline>> {
        let normalize_shader = |a: &SpecializedShader| {
            let shader = shader_store.get(&a.shader)?;
            Some(rhyolite::shader::SpecializedShader {
                stage: a.stage,
                flags: a.flags,
                shader: shader.inner().clone(),
                specialization_info: a.specialization_info.clone(),
                entry_point: a.entry_point,
            })
        };
        let shaders: Option<Vec<rhyolite::shader::SpecializedShader<'_, _>>> =
            self.shaders.iter().map(normalize_shader).collect();
        let Some(shaders) = shaders else {
            return DeferredValue::None
        };
        let layout = self.pipeline_characteristics.layout.clone();
        let create_info = self.pipeline_characteristics.create_info.clone();
        let pipeline_cache = pipeline_cache.cloned();

        let task: bevy_tasks::Task<VkResult<Arc<RayTracingPipelineLibrary>>> =
            AsyncComputeTaskPool::get().spawn(async move {
                let lib = RayTracingPipelineLibrary::create_for_shaders(
                    layout,
                    &shaders,
                    &create_info,
                    pipeline_cache.as_ref().map(|a| a.as_ref()),
                    DeferredTaskPool::get().inner().clone(),
                )
                .await?;
                tracing::trace!(handle = ?lib.raw(), "Built base pipelibe library");
                Ok(Arc::new(lib))
            });
        task.into()
    }
}

#[derive(Clone)]
pub struct RayTracingPipelineBuildInfo {
    pub pipeline_characteristics: Arc<RayTracingPipelineCharacteristics>,
    pub base_shaders: Vec<SpecializedShader>,
    pub hitgroup_shaders: Vec<(Option<SpecializedShader>, Option<SpecializedShader>, Option<SpecializedShader>, 
        RayTracingHitGroupType)>,
}
impl CachablePipeline for rhyolite::RayTracingPipeline {
    type BuildInfo = RayTracingPipelineBuildInfo;
}
impl PipelineBuildInfo for RayTracingPipelineBuildInfo {
    type Pipeline = rhyolite::RayTracingPipeline;
    fn build(
            self,
            shader_store: &Assets<ShaderModule>,
            pipeline_cache: Option<&Arc<rhyolite::PipelineCache>>,
        ) -> DeferredValue<Arc<Self::Pipeline>> {
            let layout = self.pipeline_characteristics.layout.clone();
            let normalize_shader = |a: &SpecializedShader| {
                let shader = shader_store.get(&a.shader)?;
                Some(rhyolite::shader::SpecializedShader {
                    stage: a.stage,
                    flags: a.flags,
                    shader: shader.inner().clone(),
                    specialization_info: a.specialization_info.clone(),
                    entry_point: a.entry_point,
                })
            };
            let base_shaders: Option<Vec<rhyolite::shader::SpecializedShader<'_, _>>> =
                self.base_shaders.iter().map(normalize_shader).collect();
            let Some(base_shaders) = base_shaders else {
                return DeferredValue::None;
            };
            type Shader = rhyolite::shader::SpecializedShader<'static, Arc<rhyolite::shader::ShaderModule>>;
            let hitgroups: Option<Vec<_>> = self.hitgroup_shaders.iter().map(|(rchit, rint, rahit, ty)| -> Option<(Option<Shader>, Option<Shader>, Option<Shader>, RayTracingHitGroupType)> {
                Some((
                    if let Some(rchit) = rchit.as_ref() {
                        Some(normalize_shader(rchit)?)
                    } else {
                        None
                    },if let Some(rint) = rint.as_ref() {
                        Some(normalize_shader(rint)?)
                    } else {
                        None
                    },if let Some(rahit) = rahit.as_ref() {
                        Some(normalize_shader(rahit)?)
                    } else {
                        None
                    },
                    *ty
                ))
            }).collect();
            let Some(hitgroups) = hitgroups else {
                return DeferredValue::None;
            };

            let create_info = self.pipeline_characteristics.create_info.clone();
            let pipeline_cache = pipeline_cache.cloned();
            let pipeline: bevy_tasks::Task<VkResult<Arc<RayTracingPipeline>>> =
                AsyncComputeTaskPool::get().spawn(async move {
                    let pipeline = rhyolite::RayTracingPipeline::create_for_shaders(
                        layout,
                        base_shaders.as_slice(),
                        hitgroups.into_iter(),
                        &create_info,
                        pipeline_cache.as_ref().map(|a| a.as_ref()),
                        DeferredTaskPool::get().inner().clone(),
                    )
                    .await?;
                    Ok(Arc::new(pipeline))
                });
            pipeline.into()
    }
}

impl RayTracingPipelineManager {
    pub fn layout(&self) -> &Arc<PipelineLayout> {
        &self.pipeline_characteristics.layout
    }
    pub fn new(
        pipeline_characteristics: Arc<RayTracingPipelineCharacteristics>,
        raytypes: Vec<u32>,
        raygen_shader: SpecializedShader,
        miss_shaders: Vec<SpecializedShader>,
        callable_shaders: Vec<SpecializedShader>,
    ) -> Self {
        let materials = pipeline_characteristics
            .materials
            .iter()
            .map(|_mat| RayTracingPipelineManagerMaterialInfo {
                instance_count: 0,
                pipeline_library: None,
            })
            .collect();

        let shaders = std::iter::once(raygen_shader)
            .chain(miss_shaders.into_iter())
            .chain(callable_shaders.into_iter())
            .collect();
        Self {
            raytypes,
            pipeline_base_library: None,
            pipeline_characteristics,
            current_material_flag: 0,
            specialized_pipelines: BTreeMap::new(),
            materials,
            shaders,
        }
    }
    pub fn material_instance_added<M: Material>(&mut self) {
        let id = self.pipeline_characteristics.material_to_index[&std::any::TypeId::of::<M>()];
        if self.materials[id].instance_count == 0 {
            self.current_material_flag |= 1 << id; // toggle flag
        }
        self.materials[id].instance_count += 1;
    }
    pub fn material_instance_removed<M: Material>(&mut self) {
        let id = self.pipeline_characteristics.material_to_index[&std::any::TypeId::of::<M>()];
        assert!(self.materials[id].instance_count > 0);
        self.materials[id].instance_count -= 1;
        if self.materials[id].instance_count == 0 {
            self.current_material_flag &= !(1 << id); // toggle flag
        }
    }
    pub fn get_pipeline(
        &mut self,
        pipeline_cache: &PipelineCache,
        shader_store: &Assets<ShaderModule>,
    ) -> Option<RayTracingPipelineManagerSpecializedPipeline> {
        let material_count = self.pipeline_characteristics.material_count();
        let full_material_mask = (1 << material_count) - 1;

        if !self
            .specialized_pipelines
            .contains_key(&self.current_material_flag)
        {
            self.build_specialized_pipeline(
                self.current_material_flag,
                |mat| mat.instance_count > 0,
                pipeline_cache,
            );
        }
        if full_material_mask != self.current_material_flag
            && !self.specialized_pipelines.contains_key(&full_material_mask)
        {
            self.build_specialized_pipeline(full_material_mask, |_| true, pipeline_cache);
        }


        
        if let Some(pipeline) = self.specialized_pipelines.get_mut(&self.current_material_flag).map(|ptr| unsafe {
            // Remove lifetime info here due to NLL limitation. This is safe and sound, passes polonius.
            &mut *(ptr as *mut RayTracingPipelineManagerSpecializedPipelineDeferred)
        }) {
            if let Some(p) = pipeline_cache.retrieve(&mut pipeline.pipeline, shader_store) {
                return Some(RayTracingPipelineManagerSpecializedPipeline {
                    material_mapping: &self.pipeline_characteristics.material_to_index,
                    pipeline: p,
                    hitgroup_mapping: &pipeline.hitgroup_mapping,
                    raytypes: &self.raytypes
                });
            }
        }

        if full_material_mask != self.current_material_flag && let Some(pipeline) = self.specialized_pipelines.get_mut(&full_material_mask) {
            if let Some(p) = pipeline_cache.retrieve(&mut pipeline.pipeline, shader_store) {
                tracing::trace!(material_flag = self.current_material_flag, full_material_flag = full_material_mask, "Using fallback pipeline");
                return Some(RayTracingPipelineManagerSpecializedPipeline {
                    material_mapping: &self.pipeline_characteristics.material_to_index,
                    pipeline: p,
                    hitgroup_mapping: &pipeline.hitgroup_mapping,
                    raytypes: &self.raytypes
                });
            }
        }

        None
    }
    fn build_specialized_pipeline(
        &mut self,
        material_flag: u64,
        material_filter: impl Fn(&RayTracingPipelineManagerMaterialInfo) -> bool,
        pipeline_cache: &PipelineCache,
    ) {
        self.build_specialized_pipeline_native(material_flag, material_filter, pipeline_cache);
    }
    fn build_specialized_pipeline_native(
        &mut self,
        material_flag: u64,
        material_filter: impl Fn(&RayTracingPipelineManagerMaterialInfo) -> bool,
        pipeline_cache: &PipelineCache,
    ) {
        let mut hitgroup_mapping: BTreeMap<u32, u32> = BTreeMap::new();
        let mut current_hitgroup: u32 = 0;
        let mut hitgroups = Vec::new();
        for (material_index, _material) in self
            .materials
            .iter_mut()
            .enumerate()
            .filter(|(_, material)| material_filter(&material))
        {
            hitgroup_mapping.insert(material_index as u32, current_hitgroup);
            current_hitgroup += self.raytypes.len() as u32;
            let ty = self.pipeline_characteristics.materials[material_index].ty;

            let material_hitgroups = self
                .raytypes
                .iter()
                .map(|raytype| {
                    &self.pipeline_characteristics.materials[material_index].shaders
                        [*raytype as usize]
                })
                .map(|(rchit, rint, rahit)| {
                    (rchit.clone(), rint.clone(), rahit.clone(), ty)
                });
            hitgroups.extend(material_hitgroups);
        }

        self.specialized_pipelines.insert(
            material_flag,
            RayTracingPipelineManagerSpecializedPipelineDeferred {
                pipeline: pipeline_cache.add_ray_tracing_pipeline(
                    self.pipeline_characteristics.clone(),
                    self.shaders.clone(),
                    hitgroups
                ),
                hitgroup_mapping,
            },
        );
    }

    /*
    fn build_specialized_pipeline_with_libs(
        &mut self,
        material_flag: u64,
        material_filter: impl Fn(&RayTracingPipelineManagerMaterialInfo) -> bool,
        shader_store: &Assets<ShaderModule>,
    ) {
        let mut libs: Vec<Arc<RayTracingPipelineLibrary>> =
            Vec::with_capacity(self.materials.len() + 1);

        let mut ready = true;
        if let Some(base_lib) = self.pipeline_base_library.as_mut() {
            if let Some(base_lib) = base_lib.try_get() {
                libs.push(base_lib.clone());
            } else {
                ready = false;
            };
        } else {
            // schedule build
            self.build_base_pipeline_library(shader_store);
            ready = false;
        };

        let mut hitgroup_mapping: BTreeMap<u32, u32> = BTreeMap::new();
        let mut current_hitgroup: u32 = 0;
        for (i, material) in self
            .materials
            .iter_mut()
            .enumerate()
            .filter(|(_, material)| material_filter(&material))
        {
            // For each active material
            if let Some(pipeline_library) = material.pipeline_library.as_mut() {
                if let Some(pipeline_library) = pipeline_library.try_get() {
                    libs.push(pipeline_library.clone());
                    hitgroup_mapping.insert(i as u32, current_hitgroup);
                    current_hitgroup += self.raytypes.len() as u32;
                } else {
                    // Pipeline library is being built
                    return;
                }
            } else {
                // Need to schedule build for the pipeline library.
                Self::build_material_pipeline_library(
                    i,
                    material,
                    &self.pipeline_characteristics,
                    self.pipeline_characteristics.create_info.clone(),
                    self.pipeline_cache.clone(),
                    shader_store,
                    &self.raytypes,
                );
                ready = false;
            };
        }
        if !ready {
            return;
        }
        let create_info = self.pipeline_characteristics.create_info.clone();
        let pipeline_cache = self.pipeline_cache.clone();
        let pipeline: bevy_tasks::Task<VkResult<Arc<RayTracingPipeline>>> =
            AsyncComputeTaskPool::get().spawn(async move {
                let lib = rhyolite::RayTracingPipeline::create_from_libraries(
                    libs.iter().map(|a| a.deref()),
                    &create_info,
                    pipeline_cache.as_ref().map(|a| a.as_ref()),
                    DeferredTaskPool::get().inner().clone(),
                )
                .await?;
                tracing::trace!(handle = ?lib.raw(), "Built rtx pipeline");
                drop(libs);
                drop(create_info);
                Ok(Arc::new(lib))
            });

        self.specialized_pipelines.insert(
            material_flag,
            RayTracingPipelineManagerSpecializedPipelineDeferred {
                pipeline: pipeline.into(),
                hitgroup_mapping,
            },
        );
    }
    fn build_base_pipeline_library(&mut self, shader_store: &Assets<ShaderModule>) {
        let normalize_shader = |a: &SpecializedShader| {
            let shader = shader_store.get(&a.shader)?;
            Some(rhyolite::shader::SpecializedShader {
                stage: a.stage,
                flags: a.flags,
                shader: shader.inner().clone(),
                specialization_info: a.specialization_info.clone(),
                entry_point: a.entry_point,
            })
        };
        let shaders: Option<Vec<rhyolite::shader::SpecializedShader<'_, _>>> =
            self.shaders.iter().map(normalize_shader).collect();
        let Some(shaders) = shaders else {
            return
        };
        let layout = self.pipeline_characteristics.layout.clone();
        let create_info = self.pipeline_characteristics.create_info.clone();
        let pipeline_cache = self.pipeline_cache.clone();

        let task: bevy_tasks::Task<VkResult<Arc<RayTracingPipelineLibrary>>> =
            AsyncComputeTaskPool::get().spawn(async move {
                let lib = RayTracingPipelineLibrary::create_for_shaders(
                    layout,
                    &shaders,
                    &create_info,
                    pipeline_cache.as_ref().map(|a| a.as_ref()),
                    DeferredTaskPool::get().inner().clone(),
                )
                .await?;
                tracing::trace!(handle = ?lib.raw(), "Built base pipelibe library");
                Ok(Arc::new(lib))
            });
        self.pipeline_base_library = Some(task.into());
    }
    fn build_material_pipeline_library(
        material_index: usize,
        mat: &mut RayTracingPipelineManagerMaterialInfo,
        pipeline_characteristics: &RayTracingPipelineCharacteristics,
        create_info: RayTracingPipelineLibraryCreateInfo,
        pipeline_cache: Option<Arc<PipelineCache>>,
        shader_store: &Assets<ShaderModule>,
        raytypes: &[u32],
    ) {
        let normalize_shader = |a: &SpecializedShader| {
            let shader = shader_store.get(&a.shader)?;
            Some(rhyolite::shader::SpecializedShader {
                stage: a.stage,
                flags: a.flags,
                shader: shader.inner().clone(),
                specialization_info: a.specialization_info.clone(),
                entry_point: a.entry_point,
            })
        };
        let ty = pipeline_characteristics.materials[material_index].ty;
        let hitgroups = raytypes
            .iter()
            .map(|raytype| {
                &pipeline_characteristics.materials[material_index].shaders[*raytype as usize]
            })
            .map(|(rchit, rint, rahit)| {
                let rchit = rchit.as_ref().and_then(normalize_shader);
                let rint = rint.as_ref().and_then(normalize_shader);
                let rahit = rahit.as_ref().and_then(normalize_shader);
                (rchit, rint, rahit, ty)
            })
            .collect::<Vec<_>>();
        let layout = pipeline_characteristics.layout.clone();

        let task: bevy_tasks::Task<VkResult<Arc<RayTracingPipelineLibrary>>> =
            AsyncComputeTaskPool::get().spawn(async move {
                let lib = RayTracingPipelineLibrary::create_for_hitgroups(
                    layout,
                    hitgroups.into_iter(),
                    &create_info,
                    pipeline_cache.as_ref().map(|a| a.as_ref()),
                    DeferredTaskPool::get().inner().clone(),
                )
                .await?;
                tracing::trace!(handle = ?lib.raw(), "Built material pipelibe library");
                Ok(Arc::new(lib))
            });
        mat.pipeline_library = Some(task.into());
    }
    */
}
