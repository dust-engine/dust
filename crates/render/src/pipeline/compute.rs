use std::sync::Arc;

use bevy_asset::Assets;

use bevy_tasks::AsyncComputeTaskPool;
use rhyolite::{
    ash::{prelude::VkResult, vk},
    PipelineLayout,
};

use crate::{
    deferred_task::DeferredValue, shader::SpecializedShader, CachablePipeline, PipelineBuildInfo,
    ShaderModule,
};

#[derive(Clone)]
pub struct ComputePipelineBuildInfo {
    pub layout: Arc<PipelineLayout>,
    pub shader: SpecializedShader,
}

impl CachablePipeline for rhyolite::ComputePipeline {
    type BuildInfo = ComputePipelineBuildInfo;
}
impl PipelineBuildInfo for ComputePipelineBuildInfo {
    type Pipeline = rhyolite::ComputePipeline;
    fn build(
        self,
        assets: &Assets<ShaderModule>,
        pipeline_cache: Option<&Arc<rhyolite::PipelineCache>>,
    ) -> DeferredValue<Arc<rhyolite::ComputePipeline>> {
        let Some(shader) = assets.get(&self.shader.shader) else {
            return DeferredValue::None;
        };
        let shader = shader.inner().clone();
        let cache = pipeline_cache.map(|a| a.clone());
        let pipeline: bevy_tasks::Task<VkResult<Arc<rhyolite::ComputePipeline>>> =
            AsyncComputeTaskPool::get().spawn(async move {
                let specialized_shader = rhyolite::shader::SpecializedShader {
                    stage: self.shader.stage,
                    flags: self.shader.flags,
                    shader,
                    specialization_info: self.shader.specialization_info.clone(),
                    entry_point: self.shader.entry_point,
                };
                let pipeline = rhyolite::ComputePipeline::create_with_shader_and_layout(
                    specialized_shader,
                    self.layout.clone(),
                    vk::PipelineCreateFlags::empty(),
                    cache.as_ref().map(|a| a.as_ref()),
                )?;
                Ok(Arc::new(pipeline))
            });
        pipeline.into()
    }
}
