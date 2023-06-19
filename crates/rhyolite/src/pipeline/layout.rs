use crate::{descriptor::DescriptorSetLayout, shader::ShaderModuleEntryPoint, Device, HasDevice};
use ash::{prelude::VkResult, vk};
use std::sync::Arc;

pub struct PipelineLayout {
    device: Arc<Device>,
    inner: vk::PipelineLayout,

    desc_sets: Vec<Arc<DescriptorSetLayout>>,
    push_constant_range: Vec<vk::PushConstantRange>,
}

impl PipelineLayout {
    pub fn desc_sets(&self) -> &[Arc<DescriptorSetLayout>] {
        &self.desc_sets
    }
    pub fn push_constant_range(&self) -> &[vk::PushConstantRange] {
        &self.push_constant_range
    }
    pub fn new(
        device: Arc<Device>,
        set_layouts: Vec<Arc<DescriptorSetLayout>>,
        push_constant_ranges: &[vk::PushConstantRange],
        flags: vk::PipelineLayoutCreateFlags,
    ) -> VkResult<Self> {
        let raw_set_layouts: Vec<_> = set_layouts.iter().map(|a| unsafe { a.raw() }).collect();
        let info = vk::PipelineLayoutCreateInfo {
            flags,
            set_layout_count: raw_set_layouts.len() as u32,
            p_set_layouts: raw_set_layouts.as_ptr(),
            push_constant_range_count: push_constant_ranges.len() as u32,
            p_push_constant_ranges: push_constant_ranges.as_ptr(),
            ..Default::default()
        };

        let layout = unsafe { device.create_pipeline_layout(&info, None)? };
        Ok(Self {
            device,
            inner: layout,
            desc_sets: set_layouts,
            push_constant_range: Vec::new(),
        })
    }
    /// Create pipeline layout for pipelines with only one shader entry point. Only applicable to compute shaders.
    pub fn for_layout(
        device: Arc<Device>,
        entry_point: ShaderModuleEntryPoint,
        flags: vk::PipelineLayoutCreateFlags,
    ) -> VkResult<Self> {
        let set_layouts: Vec<_> = entry_point
            .desc_sets
            .iter()
            .map(|a| unsafe { a.raw() })
            .collect();
        let mut info = vk::PipelineLayoutCreateInfo {
            flags,
            set_layout_count: set_layouts.len() as u32,
            p_set_layouts: set_layouts.as_ptr(),
            ..Default::default()
        };
        if let Some(push_constant_range) = entry_point.push_constant_range.as_ref() {
            info.push_constant_range_count = 1;
            info.p_push_constant_ranges = push_constant_range;
        }
        let layout = unsafe { device.create_pipeline_layout(&info, None)? };
        Ok(Self {
            device,
            inner: layout,
            desc_sets: entry_point.desc_sets,
            push_constant_range: if let Some(range) = entry_point.push_constant_range {
                vec![range]
            } else {
                Vec::new()
            },
        })
    }
    pub unsafe fn raw(&self) -> vk::PipelineLayout {
        self.inner
    }
}
impl HasDevice for PipelineLayout {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}
impl Drop for PipelineLayout {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline_layout(self.inner, None);
        }
    }
}
