use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
};

use bevy_app::Plugin;
use bevy_asset::{AssetEvent, Assets, Handle};
use bevy_ecs::{
    prelude::EventReader,
    system::{ResMut, Resource},
    world::FromWorld,
};
use rhyolite::{ComputePipeline, PipelineLayout, RayTracingHitGroupType};

use crate::{
    deferred_task::DeferredValue, ComputePipelineBuildInfo, ShaderModule, SpecializedShader, RayTracingPipelineCharacteristics,
};

use super::manager::RayTracingPipelineBuildInfo;

#[derive(Resource)]
pub struct PipelineCache {
    cache: Option<Arc<rhyolite::PipelineCache>>,
    shader_generations: HashMap<Handle<ShaderModule>, u32>,
    hot_reload_enabled: bool,
}

pub struct CachedPipeline<T: CachablePipeline> {
    build_info: Option<T::BuildInfo>,
    pipeline: Option<Arc<T>>,
    task: DeferredValue<Arc<T>>,
    shader_generations: HashMap<Handle<ShaderModule>, u32>,
}

pub trait CachablePipeline {
    type BuildInfo: PipelineBuildInfo<Pipeline = Self>;
}

pub trait PipelineBuildInfo: Clone {
    type Pipeline: CachablePipeline<BuildInfo = Self>;
    fn build(
        self,
        assets: &Assets<ShaderModule>,
        cache: Option<&Arc<rhyolite::PipelineCache>>,
    ) -> DeferredValue<Arc<Self::Pipeline>>;
}

impl<T: CachablePipeline> CachedPipeline<T> {
    pub fn is_ready(&self) -> bool {
        self.task.is_done()
    }
}

impl PipelineCache {
    pub fn add_compute_pipeline(
        &self,
        layout: Arc<PipelineLayout>,
        shader: SpecializedShader,
    ) -> CachedPipeline<ComputePipeline> {
        CachedPipeline {
            shader_generations: if self.hot_reload_enabled {
                let mut map = HashMap::new();
                map.insert(shader.shader.clone_weak(), 0);
                map
            } else {
                Default::default()
            },
            build_info: Some(ComputePipelineBuildInfo { layout, shader }),
            pipeline: None,
            task: DeferredValue::None,
        }
    }
    pub fn add_ray_tracing_pipeline(
        &self,
        pipeline_characteristics: Arc<RayTracingPipelineCharacteristics>,
        base_shaders: Vec<SpecializedShader>,
        hitgroup_shaders: Vec<(Option<SpecializedShader>, Option<SpecializedShader>, Option<SpecializedShader>, RayTracingHitGroupType)>,
    ) -> CachedPipeline<rhyolite::RayTracingPipeline> {
        CachedPipeline {
            
            shader_generations: if self.hot_reload_enabled {
                base_shaders.iter()
                .chain(hitgroup_shaders.iter().flat_map(|(rchit, rint, rahit, _)| {
                    rchit.into_iter().chain(rint).chain(rahit)
                }))
                .map(|shader| (shader.shader.clone_weak(), 0))
                .collect()
            } else {
                Default::default()
            },
            build_info: Some(
                RayTracingPipelineBuildInfo {
                    pipeline_characteristics,
                    base_shaders,
                    hitgroup_shaders,
                }
            ),
            pipeline: None,
            task: DeferredValue::None,
        }
    }
    pub fn retrieve<'a, T: CachablePipeline>(
        &self,
        cached_pipeline: &'a mut CachedPipeline<T>,
        assets: &Assets<ShaderModule>,
    ) -> Option<&'a Arc<T>> {
        if let Some(pipeline) = cached_pipeline.task.take() {
            cached_pipeline.pipeline = Some(pipeline);
        }

        if self.hot_reload_enabled {
            for (shader, generation) in cached_pipeline.shader_generations.iter() {
                if let Some(latest_generation) = self.shader_generations.get(shader) {
                    if latest_generation > generation {
                        // schedule.
                        cached_pipeline.task = cached_pipeline
                            .build_info
                            .as_ref()
                            .unwrap()
                            .clone()
                            .build(assets, self.cache.as_ref());
                        if !cached_pipeline.task.is_none() {
                            // If a new shader build task was successfully scheduled, update all shader generations
                            // to latest so that this particular pipeline won't be updated next frame.
                            // Otherwise, leave it as is, and this is going to attempt creating a new pipeline
                            // compilation task again next time.
                            for (shader, generation) in cached_pipeline.shader_generations.iter_mut() {
                                if let Some(latest_generation) = self.shader_generations.get(shader) {
                                    *generation = *latest_generation;
                                }
                            }
                            tracing::info!("Shader hot reload: updated");
                        }
                        // TODO: what if this returns None? will this be invoked multiple times? due to generation not getting updated
                        break;
                    }
                }
            }
        }

        if let Some(pipeline) = cached_pipeline.pipeline.as_ref() {
            return Some(pipeline);
        } else {
            if cached_pipeline.task.is_none() {
                // schedule
                if self.hot_reload_enabled {
                    cached_pipeline.task = cached_pipeline
                        .build_info
                        .as_ref()
                        .unwrap()
                        .clone()
                        .build(assets, self.cache.as_ref());
                } else {
                    cached_pipeline.task = cached_pipeline
                        .build_info
                        .take()
                        .unwrap()
                        .build(assets, self.cache.as_ref());
                }
            }
            return None;
        }
    }
}

fn pipeline_cache_shader_updated_system(
    mut pipeline_cache: ResMut<PipelineCache>,
    mut events: EventReader<AssetEvent<ShaderModule>>,
) {
    for event in events.iter() {
        match event {
            AssetEvent::Created { handle } => (),
            AssetEvent::Modified { handle } => {
                let generation = pipeline_cache
                    .shader_generations
                    .entry(handle.clone_weak())
                    .or_default();
                *generation += 1;
            }
            AssetEvent::Removed { handle } => {
                pipeline_cache.shader_generations.remove(handle);
            }
        }
    }
}

pub struct PipelineCachePlugin {
    shader_hot_reload: bool,
    pipeline_cache_enabled: bool, // TODO: use pipeline cache
}

impl Default for PipelineCachePlugin {
    fn default() -> Self {
        Self {
            shader_hot_reload: true,
            pipeline_cache_enabled: false,
        }
    }
}

impl Plugin for PipelineCachePlugin {
    fn build(&self, app: &mut bevy_app::App) {
        let cache = PipelineCache {
            cache: None, // TODO
            shader_generations: Default::default(),
            hot_reload_enabled: self.shader_hot_reload,
        };
        app.insert_resource(cache);
        if self.shader_hot_reload {
            app.add_systems(bevy_app::PreUpdate, pipeline_cache_shader_updated_system);
        }
    }
}
