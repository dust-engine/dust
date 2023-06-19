use ash::{prelude::VkResult, vk};

use super::PipelineCache;
use crate::{
    shader::{ShaderModule, SpecializedReflectedShader, SpecializedShader},
    HasDevice,
};

use super::PipelineLayout;
use std::{ops::Deref, sync::Arc};

pub struct ComputePipeline {
    layout: Arc<PipelineLayout>,
    pipeline: vk::Pipeline,
}
impl ComputePipeline {
    pub fn layout(&self) -> &Arc<PipelineLayout> {
        &self.layout
    }
    pub unsafe fn raw(&self) -> vk::Pipeline {
        self.pipeline
    }
    pub unsafe fn raw_layout(&self) -> vk::PipelineLayout {
        self.layout.raw()
    }
}
impl HasDevice for ComputePipeline {
    fn device(&self) -> &Arc<crate::Device> {
        self.layout.device()
    }
}
impl Drop for ComputePipeline {
    fn drop(&mut self) {
        unsafe { self.layout.device().destroy_pipeline(self.pipeline, None) }
    }
}
pub struct ComputePipelineCreateInfo<'a> {
    pub pipeline_layout_create_flags: vk::PipelineLayoutCreateFlags,
    pub pipeline_create_flags: vk::PipelineCreateFlags,
    pub pipeline_cache: Option<&'a PipelineCache>,
}

impl<'a> Default for ComputePipelineCreateInfo<'a> {
    fn default() -> Self {
        Self {
            pipeline_layout_create_flags: vk::PipelineLayoutCreateFlags::empty(),
            pipeline_create_flags: vk::PipelineCreateFlags::empty(),
            pipeline_cache: None,
        }
    }
}

impl ComputePipeline {
    pub fn create_with_reflected_shader<'a>(
        shader: SpecializedReflectedShader<'a>,
        info: ComputePipelineCreateInfo<'a>,
    ) -> VkResult<Self> {
        let layout = PipelineLayout::for_layout(
            shader.device().clone(),
            shader.entry_point().clone(),
            info.pipeline_layout_create_flags,
        )?;
        Self::create_with_shader_and_layout(
            shader.into(),
            Arc::new(layout),
            info.pipeline_create_flags,
            info.pipeline_cache,
        )
    }
    pub fn create_with_shader_and_layout<'a, S: Deref<Target = ShaderModule>>(
        shader: SpecializedShader<'a, S>,
        layout: Arc<PipelineLayout>,
        pipeline_create_flags: vk::PipelineCreateFlags,
        pipeline_cache: Option<&'a PipelineCache>,
    ) -> VkResult<Self> {
        let device = shader.device().clone();
        let pipeline = unsafe {
            let mut pipeline = vk::Pipeline::null();
            let specialization_info = shader.specialization_info.raw_info();
            (device.fp_v1_0().create_compute_pipelines)(
                device.handle(),
                pipeline_cache.map(|a| a.raw()).unwrap_or_default(),
                1,
                &vk::ComputePipelineCreateInfo {
                    flags: pipeline_create_flags,
                    stage: vk::PipelineShaderStageCreateInfo {
                        flags: shader.flags,
                        stage: vk::ShaderStageFlags::COMPUTE,
                        module: shader.shader.raw(),
                        p_name: shader.entry_point.as_ptr(),
                        p_specialization_info: &specialization_info,
                        ..Default::default()
                    },
                    layout: layout.raw(),
                    // Do not use pipeline derivative as they're not beneficial.
                    // https://stackoverflow.com/questions/37135130/vulkan-creating-and-benefit-of-pipeline-derivatives
                    base_pipeline_handle: vk::Pipeline::null(),
                    base_pipeline_index: 0,
                    ..Default::default()
                },
                std::ptr::null(),
                (&mut pipeline) as *mut _,
            )
            .result_with_success(pipeline)
        }?;
        Ok(Self { layout, pipeline })
    }
}
