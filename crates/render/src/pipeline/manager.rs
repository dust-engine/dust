use std::{
    collections::{BTreeMap, HashMap},
    ops::Deref,
    sync::Arc,
};

use bevy_asset::Assets;
use bevy_tasks::AsyncComputeTaskPool;
use rhyolite::{
    ash::prelude::VkResult, HasDevice, PipelineCache, PipelineLayout, RayTracingPipeline,
    RayTracingPipelineLibrary, RayTracingPipelineLibraryCreateInfo, Task,
};

use crate::{
    deferred_task::{DeferredTaskPool, DeferredValue},
    material::Material,
    shader::{ShaderModule, SpecializedShader},
};

use super::RayTracingPipelineCharacteristics;

struct RayTracingPipelineManagerMaterialInfo {
    instance_count: usize,
    pipeline_library: Option<DeferredValue<Arc<RayTracingPipelineLibrary>>>,
}
struct RayTracingPipelineManagerSpecializedPipelineDeferred {
    pipeline: DeferredValue<Arc<rhyolite::RayTracingPipeline>>,
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
        let hitgroup_index = self.hitgroup_mapping[&material_index] + raytype;
        self.pipeline
            .sbt_handles()
            .hitgroup(hitgroup_index as usize)
    }
}

pub struct RayTracingPipelineManager {
    /// A pipeline library containing raygen, raymiss, callable shaders
    pipeline_base_library: Option<DeferredValue<Arc<RayTracingPipelineLibrary>>>,
    pipeline_characteristics: Arc<RayTracingPipelineCharacteristics>,
    current_material_flag: u64,
    specialized_pipelines: BTreeMap<u64, RayTracingPipelineManagerSpecializedPipelineDeferred>,
    materials: Vec<RayTracingPipelineManagerMaterialInfo>,

    pipeline_cache: Option<Arc<PipelineCache>>,

    shaders: Vec<SpecializedShader>,
}

impl RayTracingPipelineManager {
    pub fn layout(&self) -> &Arc<PipelineLayout> {
        &self.pipeline_characteristics.layout
    }
    pub fn new(
        pipeline_characteristics: Arc<RayTracingPipelineCharacteristics>,
        raygen_shader: SpecializedShader,
        miss_shaders: Vec<SpecializedShader>,
        callable_shaders: Vec<SpecializedShader>,
        pipeline_cache: Option<Arc<PipelineCache>>,
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
            pipeline_base_library: None,
            pipeline_characteristics,
            current_material_flag: 0,
            specialized_pipelines: BTreeMap::new(),
            materials,
            pipeline_cache,
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
        shader_store: &Assets<ShaderModule>,
    ) -> Option<RayTracingPipelineManagerSpecializedPipeline> {
        let material_count = self.pipeline_characteristics.material_count();
        let full_material_mask = (1 << material_count) - 1;

        if let Some(pipeline) = self.specialized_pipelines.get(&self.current_material_flag) &&
        pipeline.pipeline.is_done()  {
            let pipeline = self.specialized_pipelines.get_mut(&self.current_material_flag).unwrap();
            let p = pipeline.pipeline.try_get().unwrap();
            Some(RayTracingPipelineManagerSpecializedPipeline {
                material_mapping: &self.pipeline_characteristics.material_to_index,
                pipeline: p,
                hitgroup_mapping: &pipeline.hitgroup_mapping
            })
        } else if let Some(pipeline) = self.specialized_pipelines.get(&full_material_mask) &&
        pipeline.pipeline.is_done() {
            // Fallback to general purpose pipeline
            self.build_specialized_pipeline(self.current_material_flag, |mat| mat.instance_count > 0, shader_store);
            let pipeline = self.specialized_pipelines.get_mut(&self.current_material_flag).unwrap();
            let p = pipeline.pipeline.try_get().unwrap();
            Some(RayTracingPipelineManagerSpecializedPipeline {
                material_mapping: &self.pipeline_characteristics.material_to_index,
                pipeline: p,
                hitgroup_mapping: &pipeline.hitgroup_mapping
            })
        } else {
            // Even the full featured pipeline wasn't built yet.
            self.build_specialized_pipeline(full_material_mask, |_| true, shader_store);
            None
        }
    }
    fn build_specialized_pipeline(
        &mut self,
        material_flag: u64,
        material_filter: impl Fn(&RayTracingPipelineManagerMaterialInfo) -> bool,
        shader_store: &Assets<ShaderModule>,
    ) {
        let mut libs: Vec<Arc<RayTracingPipelineLibrary>> =
            Vec::with_capacity(self.materials.len() + 1);

        let Some(base_lib) = self.pipeline_base_library.as_mut() else {
            // schedule build
            self.build_base_pipeline_library(shader_store);
            return;
        };
        let Some(base_lib) = base_lib.try_get() else {
            return;
        };
        println!("Discovered bulit base library: {:?}", base_lib.raw());
        libs.push(base_lib.clone());

        let mut hitgroup_mapping: BTreeMap<u32, u32> = BTreeMap::new();
        let mut current_hitgroup: u32 = 0;
        for (i, material) in self
            .materials
            .iter_mut()
            .enumerate()
            .filter(|(_, material)| material_filter(&material))
        {
            // For each active material
            let Some(pipeline_library) = material.pipeline_library.as_mut() else {
                // Need to schedule build for the pipeline library.
                Self::build_material_pipeline_library(i, material, &self.pipeline_characteristics, self.pipeline_characteristics.create_info.clone(), self.pipeline_cache.clone(), shader_store);
                return;
            };
            let Some(pipeline_library) = pipeline_library.try_get() else {
                // Pipeline library is being built
                return;
            };
            println!("Discovered bulit library: {:?}", pipeline_library.raw());
            libs.push(pipeline_library.clone());
            hitgroup_mapping.insert(i as u32, current_hitgroup);
            current_hitgroup += self.pipeline_characteristics.num_raytype;
        }
        let create_info = self.pipeline_characteristics.create_info.clone();
        let pipeline: bevy_tasks::Task<VkResult<Arc<RayTracingPipeline>>> =
            AsyncComputeTaskPool::get().spawn(async move {
                let lib = rhyolite::RayTracingPipeline::create_from_libraries(
                    libs.iter().map(|a| a.deref()),
                    &create_info,
                    None,
                    DeferredTaskPool::get().inner().clone(),
                )
                .await?;
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
        let hitgroups = pipeline_characteristics.materials[material_index]
            .shaders
            .iter()
            .map(|(rchit, rint, rahit)| {
                let rchit = rchit.as_ref().and_then(normalize_shader);
                let rint = rint.as_ref().and_then(normalize_shader);
                let rahit = rahit.as_ref().and_then(normalize_shader);
                (rchit, rint, rahit)
            })
            .collect::<Vec<_>>();
        let layout = pipeline_characteristics.layout.clone();

        let task: bevy_tasks::Task<VkResult<Arc<RayTracingPipelineLibrary>>> =
            AsyncComputeTaskPool::get().spawn(async move {
                let lib = RayTracingPipelineLibrary::create_for_hitgroups(
                    layout,
                    hitgroups.into_iter(),
                    &create_info,
                    ty,
                    pipeline_cache.as_ref().map(|a| a.as_ref()),
                    DeferredTaskPool::get().inner().clone(),
                )
                .await?;
                Ok(Arc::new(lib))
            });
        mat.pipeline_library = Some(task.into());
    }
}
