use crate::Device;
use ash::{prelude::VkResult, vk};
use std::{collections::BTreeMap, sync::Arc};

pub struct DescriptorSetLayout {
    device: Arc<Device>,
    pub(crate) raw: vk::DescriptorSetLayout,
    pub(crate) desc_types: Vec<(vk::DescriptorType, u32)>,
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
    /// TODO: Actually cache this, or not.
    pub fn new(
        device: Arc<Device>,
        binding_infos: &[vk::DescriptorSetLayoutBinding],
        flags: vk::DescriptorSetLayoutCreateFlags,
    ) -> VkResult<Self> {
        let raw = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo {
                    flags,
                    binding_count: binding_infos.len() as u32,
                    p_bindings: binding_infos.as_ptr(),
                    ..Default::default()
                },
                None,
            )
        }?;

        let mut desc_types = BTreeMap::new();

        for binding in binding_infos.iter() {
            if binding.p_immutable_samplers.is_null() {
                let count = desc_types.entry(binding.descriptor_type).or_insert(0);
                if binding.descriptor_type == vk::DescriptorType::INLINE_UNIFORM_BLOCK {
                    // We take the next multiple of 8 here because on AMD, descriptor pool allocations seem
                    // to be aligned to the 8 byte boundary. See
                    // https://gist.github.com/Neo-Zhixing/992a0e789e34b59285026dd8161b9112
                    *count += binding.descriptor_count.next_multiple_of(8);
                } else {
                    *count += binding.descriptor_count;
                }
            } else {
                if binding.descriptor_type == vk::DescriptorType::COMBINED_IMAGE_SAMPLER {
                    let count = desc_types.entry(binding.descriptor_type).or_insert(0);
                    *count += binding.descriptor_count;
                } else {
                    // Don't need separate descriptor if the sampler was built into the layout
                    assert_eq!(binding.descriptor_type, vk::DescriptorType::SAMPLER);
                }
            }
        }

        Ok(Self {
            device,
            raw,
            desc_types: desc_types.into_iter().collect(),
        })
    }
    pub unsafe fn raw(&self) -> vk::DescriptorSetLayout {
        self.raw
    }
}
