use crate::{sampler::Sampler, Device, HasDevice};
use ash::{prelude::VkResult, vk};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct DescriptorSetLayoutBindingInfo {
    pub binding: u32,
    pub descriptor_type: vk::DescriptorType,
    pub descriptor_count: u32,
    pub stage_flags: vk::ShaderStageFlags,
    pub immutable_samplers: Vec<Arc<Sampler>>,
}
impl std::hash::Hash for DescriptorSetLayoutBindingInfo {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.binding.hash(state);
        self.descriptor_type.hash(state);
        self.descriptor_count.hash(state);
        self.stage_flags.hash(state);
        for sampler in self.immutable_samplers.iter() {
            unsafe { sampler.raw() }.hash(state);
        }
    }
}
impl PartialEq for DescriptorSetLayoutBindingInfo {
    fn eq(&self, other: &Self) -> bool {
        self.binding == other.binding
            && self.descriptor_type == other.descriptor_type
            && self.descriptor_count == other.descriptor_count
            && self.stage_flags == other.stage_flags
            && self.immutable_samplers.len() == other.immutable_samplers.len()
            && self
                .immutable_samplers
                .iter()
                .zip(other.immutable_samplers.iter())
                .all(|(a, b)| unsafe { a.raw() == b.raw() })
    }
}
impl Eq for DescriptorSetLayoutBindingInfo {}

pub struct DescriptorSetLayout {
    device: Arc<Device>,
    raw: vk::DescriptorSetLayout,
    pub binding_infos: Vec<DescriptorSetLayoutBindingInfo>,
}
impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_descriptor_set_layout(self.raw, None);
        }
    }
}
impl DescriptorSetLayout {
    /// Users should obtain the layout from the cache.
    fn new(
        device: Arc<Device>,
        binding_infos: Vec<DescriptorSetLayoutBindingInfo>,
        flags: vk::DescriptorSetLayoutCreateFlags,
    ) -> VkResult<Self> {
        let raw = unsafe {
            let total_immutable_samplers = binding_infos
                .iter()
                .map(|a| a.immutable_samplers.len())
                .sum();
            let mut immutable_samplers: Vec<vk::Sampler> =
                Vec::with_capacity(total_immutable_samplers);
            let bindings: Vec<_> = binding_infos
                .iter()
                .map(|binding| {
                    let immutable_samplers_offset = immutable_samplers.len();
                    immutable_samplers.extend(binding.immutable_samplers.iter().map(|a| a.raw()));
                    if binding.immutable_samplers.len() > 0 {
                        assert_eq!(
                            binding.immutable_samplers.len() as u32,
                            binding.descriptor_count
                        );
                    }
                    vk::DescriptorSetLayoutBinding {
                        binding: binding.binding,
                        descriptor_type: binding.descriptor_type,
                        descriptor_count: binding.descriptor_count,
                        stage_flags: binding.stage_flags,
                        p_immutable_samplers: if binding.immutable_samplers.is_empty() {
                            std::ptr::null()
                        } else {
                            immutable_samplers.as_ptr().add(immutable_samplers_offset)
                        },
                    }
                })
                .collect();
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo {
                    flags,
                    binding_count: binding_infos.len() as u32,
                    p_bindings: bindings.as_ptr(),
                    ..Default::default()
                },
                None,
            )
        }?;
        Ok(Self {
            device,
            binding_infos,
            raw,
        })
    }
    pub unsafe fn raw(&self) -> vk::DescriptorSetLayout {
        self.raw
    }
}

#[derive(PartialEq, Eq, Hash, Debug)]
pub struct DescriptorSetLayoutCacheKey {
    pub bindings: Vec<DescriptorSetLayoutBindingInfo>,
    pub flags: vk::DescriptorSetLayoutCreateFlags,
}
impl DescriptorSetLayoutCacheKey {
    pub fn build(self, device: Arc<Device>) -> VkResult<DescriptorSetLayout> {
        DescriptorSetLayout::new(device, self.bindings, self.flags)
    }
}
