use ash::{prelude::VkResult, vk};
use std::{collections::BTreeMap, ffi::c_void, sync::Arc};

use crate::{Device, HasDevice, PipelineLayout};

pub struct DescriptorPool {
    device: Arc<Device>,
    pool: vk::DescriptorPool,
}
impl Drop for DescriptorPool {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_descriptor_pool(self.pool, None);
        }
    }
}
impl HasDevice for DescriptorPool {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl DescriptorPool {
    pub fn allocate_for_pipeline_layout(
        &mut self,
        pipeline: &PipelineLayout,
    ) -> VkResult<Vec<vk::DescriptorSet>> {
        let set_layouts: Vec<_> = pipeline
            .desc_sets()
            .iter()
            .map(|a| unsafe { a.raw() })
            .collect();
        let info = vk::DescriptorSetAllocateInfo {
            descriptor_pool: self.pool,
            descriptor_set_count: set_layouts.len() as u32,
            p_set_layouts: set_layouts.as_ptr(),
            ..Default::default()
        };
        unsafe { self.device.allocate_descriptor_sets(&info) }
    }
    /// Create a descriptor pool just big enough to accommodate one of each pipeline layout,
    /// for `multiplier` times. This is useful when you have multiple pipeline layouts, each
    /// having distinct descriptor layouts and bindings. `multiplier` would generally match the
    /// max number of frames in flight.
    pub fn for_pipeline_layouts<T: std::ops::Deref<Target = PipelineLayout>>(
        layouts: impl IntoIterator<Item = T>,
        multiplier: u32,
    ) -> VkResult<Self> {
        let mut desc_types: BTreeMap<vk::DescriptorType, u32> = BTreeMap::new();
        let mut max_sets: u32 = 0;
        let mut device: Option<Arc<Device>> = None;

        let mut inline_uniform_block_create_info =
            vk::DescriptorPoolInlineUniformBlockCreateInfo::default();
        for pipeline_layout in layouts.into_iter() {
            let pipeline_layout = pipeline_layout.deref();
            max_sets += pipeline_layout.desc_sets().len() as u32;
            if let Some(device) = device.as_ref() {
                assert!(Arc::ptr_eq(device, pipeline_layout.device()));
            } else {
                device.replace(pipeline_layout.device().clone());
            }
            for desc_set_layout in pipeline_layout.desc_sets().iter() {
                for binding in desc_set_layout.binding_infos.iter() {
                    if binding.immutable_samplers.is_empty() {
                        let count = desc_types.entry(binding.descriptor_type).or_insert(0);
                        if binding.descriptor_type == vk::DescriptorType::INLINE_UNIFORM_BLOCK {
                            // We take the next multiple of 8 here because on AMD, descriptor pool allocations seem
                            // to be aligned to the 8 byte boundary. See
                            // https://gist.github.com/Neo-Zhixing/992a0e789e34b59285026dd8161b9112
                            *count += binding.descriptor_count.next_multiple_of(8);
                            inline_uniform_block_create_info.max_inline_uniform_block_bindings +=
                                multiplier;
                        } else {
                            *count += binding.descriptor_count;
                        }
                    } else {
                        // Don't need separate descriptor if the sampler was built into the layout.
                        // TODO: combined image samplers: need to allocate still?
                        assert_eq!(binding.descriptor_type, vk::DescriptorType::SAMPLER);
                        assert_eq!(
                            binding.immutable_samplers.len() as u32,
                            binding.descriptor_count
                        );
                    }
                }
            }
        }
        let pool_sizes: Vec<_> = desc_types
            .into_iter()
            .map(|(ty, descriptor_count)| vk::DescriptorPoolSize {
                ty,
                descriptor_count: descriptor_count * multiplier,
            })
            .collect();
        let mut info = vk::DescriptorPoolCreateInfo {
            max_sets: max_sets * multiplier,
            p_pool_sizes: pool_sizes.as_ptr(),
            pool_size_count: pool_sizes.len() as u32,
            ..Default::default()
        };
        if inline_uniform_block_create_info.max_inline_uniform_block_bindings > 0 {
            info.p_next = &mut inline_uniform_block_create_info as *mut _ as *mut c_void;
        }
        let device = device.expect("Expects at least one pipeline layout.");
        let pool = unsafe { device.create_descriptor_pool(&info, None)? };
        Ok(Self { device, pool })
    }
}

pub struct DescriptorSet {
    device: Arc<Device>,
    set: vk::DescriptorSet,
}

pub struct DescriptorSetWrite<'a> {
    pub dst_set: vk::DescriptorSet,
    pub dst_binding: u32,
    pub dst_array_element: u32,
    ty: DescriptorSetWriteType<'a>,
}
impl<'a> DescriptorSetWrite<'a> {
    pub fn input_attachments(
        set: vk::DescriptorSet,
        binding: u32,
        array_element: u32,
        images: &'a [vk::DescriptorImageInfo],
    ) -> Self {
        Self {
            dst_set: set,
            dst_binding: binding,
            dst_array_element: array_element,
            ty: DescriptorSetWriteType::InputAttachment(images),
        }
    }
    pub fn combined_image_samplers(
        set: vk::DescriptorSet,
        binding: u32,
        array_element: u32,
        images: &'a [vk::DescriptorImageInfo],
    ) -> Self {
        Self {
            dst_set: set,
            dst_binding: binding,
            dst_array_element: array_element,
            ty: DescriptorSetWriteType::CombinedImageSampler(images),
        }
    }
    pub fn storage_images(
        set: vk::DescriptorSet,
        binding: u32,
        array_element: u32,
        images: &'a [vk::DescriptorImageInfo],
    ) -> Self {
        Self {
            dst_set: set,
            dst_binding: binding,
            dst_array_element: array_element,
            ty: DescriptorSetWriteType::StorageImage(images),
        }
    }
    pub fn sampled_images(
        set: vk::DescriptorSet,
        binding: u32,
        array_element: u32,
        images: &'a [vk::DescriptorImageInfo],
    ) -> Self {
        Self {
            dst_set: set,
            dst_binding: binding,
            dst_array_element: array_element,
            ty: DescriptorSetWriteType::SampledImage(images),
        }
    }
    pub fn samplers(
        set: vk::DescriptorSet,
        binding: u32,
        array_element: u32,
        samplers: &'a [vk::DescriptorImageInfo],
    ) -> Self {
        Self {
            dst_set: set,
            dst_binding: binding,
            dst_array_element: array_element,
            ty: DescriptorSetWriteType::Sampler(samplers),
        }
    }
    pub fn uniform_buffers(
        set: vk::DescriptorSet,
        binding: u32,
        array_element: u32,
        buffers: &'a [vk::DescriptorBufferInfo],
        with_dynamic_offset: bool,
    ) -> Self {
        Self {
            dst_set: set,
            dst_binding: binding,
            dst_array_element: array_element,
            ty: if with_dynamic_offset {
                DescriptorSetWriteType::UniformBufferDynamic(buffers)
            } else {
                DescriptorSetWriteType::UniformBuffer(buffers)
            },
        }
    }
    pub fn storage_buffers(
        set: vk::DescriptorSet,
        binding: u32,
        array_element: u32,
        buffers: &'a [vk::DescriptorBufferInfo],
        with_dynamic_offset: bool,
    ) -> Self {
        Self {
            dst_set: set,
            dst_binding: binding,
            dst_array_element: array_element,
            ty: if with_dynamic_offset {
                DescriptorSetWriteType::StorageBufferDynamic(buffers)
            } else {
                DescriptorSetWriteType::StorageBuffer(buffers)
            },
        }
    }
    pub fn accel_structs(
        set: vk::DescriptorSet,
        binding: u32,
        array_element: u32,
        accel_structs: &'a [vk::AccelerationStructureKHR],
    ) -> Self {
        Self {
            dst_set: set,
            dst_binding: binding,
            dst_array_element: array_element,
            ty: DescriptorSetWriteType::AccelerationStructure {
                info: vk::WriteDescriptorSetAccelerationStructureKHR {
                    p_acceleration_structures: accel_structs.as_ptr(),
                    acceleration_structure_count: accel_structs.len() as u32,
                    ..Default::default()
                },
                accel_structs,
            },
        }
    }
    pub fn inline_uniform_block(
        set: vk::DescriptorSet,
        binding: u32,
        array_element: u32,
        data: &'a [u32],
    ) -> Self {
        Self {
            dst_set: set,
            dst_binding: binding,
            dst_array_element: array_element,
            ty: DescriptorSetWriteType::InlineUniformBlock {
                info: vk::WriteDescriptorSetInlineUniformBlock {
                    p_data: data.as_ptr() as *const _,
                    data_size: std::mem::size_of_val(data) as u32,
                    ..Default::default()
                },
                data,
            },
        }
    }
}
pub enum DescriptorSetWriteType<'a> {
    Sampler(&'a [vk::DescriptorImageInfo]),
    CombinedImageSampler(&'a [vk::DescriptorImageInfo]),
    SampledImage(&'a [vk::DescriptorImageInfo]),
    StorageImage(&'a [vk::DescriptorImageInfo]),

    UniformTexelBuffer(&'a [vk::BufferView]),
    StorageTexelBuffer(&'a [vk::BufferView]),

    UniformBuffer(&'a [vk::DescriptorBufferInfo]),
    StorageBuffer(&'a [vk::DescriptorBufferInfo]),
    UniformBufferDynamic(&'a [vk::DescriptorBufferInfo]),
    StorageBufferDynamic(&'a [vk::DescriptorBufferInfo]),

    InputAttachment(&'a [vk::DescriptorImageInfo]),

    InlineUniformBlock {
        info: vk::WriteDescriptorSetInlineUniformBlock,
        data: &'a [u32],
    },

    AccelerationStructure {
        info: vk::WriteDescriptorSetAccelerationStructureKHR,
        accel_structs: &'a [vk::AccelerationStructureKHR],
    },
}

impl Device {
    pub fn write_descriptor_sets<const N: usize>(&self, mut writes: [DescriptorSetWrite; N]) {
        let mut write_results: [vk::WriteDescriptorSet; N] = [Default::default(); N];
        for (i, a) in writes.iter_mut().enumerate() {
            let mut write = vk::WriteDescriptorSet {
                dst_set: a.dst_set,
                dst_binding: a.dst_binding,
                dst_array_element: a.dst_array_element,
                descriptor_type: match &a.ty {
                    DescriptorSetWriteType::Sampler(_) => vk::DescriptorType::SAMPLER,
                    DescriptorSetWriteType::CombinedImageSampler(_) => {
                        vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                    }
                    DescriptorSetWriteType::SampledImage(_) => vk::DescriptorType::SAMPLED_IMAGE,
                    DescriptorSetWriteType::StorageImage(_) => vk::DescriptorType::STORAGE_IMAGE,
                    DescriptorSetWriteType::UniformTexelBuffer(_) => {
                        vk::DescriptorType::UNIFORM_TEXEL_BUFFER
                    }
                    DescriptorSetWriteType::StorageTexelBuffer(_) => {
                        vk::DescriptorType::STORAGE_TEXEL_BUFFER
                    }
                    DescriptorSetWriteType::UniformBuffer(_) => vk::DescriptorType::UNIFORM_BUFFER,
                    DescriptorSetWriteType::StorageBuffer(_) => vk::DescriptorType::STORAGE_BUFFER,
                    DescriptorSetWriteType::UniformBufferDynamic(_) => {
                        vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                    }
                    DescriptorSetWriteType::StorageBufferDynamic(_) => {
                        vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
                    }
                    DescriptorSetWriteType::InputAttachment(_) => {
                        vk::DescriptorType::INPUT_ATTACHMENT
                    }
                    DescriptorSetWriteType::InlineUniformBlock { .. } => {
                        vk::DescriptorType::INLINE_UNIFORM_BLOCK
                    }
                    DescriptorSetWriteType::AccelerationStructure { .. } => {
                        vk::DescriptorType::ACCELERATION_STRUCTURE_KHR
                    }
                },
                ..Default::default()
            };
            match &mut a.ty {
                DescriptorSetWriteType::StorageImage(image_writes)
                | DescriptorSetWriteType::SampledImage(image_writes)
                | DescriptorSetWriteType::CombinedImageSampler(image_writes)
                | DescriptorSetWriteType::InputAttachment(image_writes)
                | DescriptorSetWriteType::Sampler(image_writes) => {
                    write.descriptor_count = image_writes.len() as u32;
                    write.p_image_info = (*image_writes).as_ptr();
                }
                DescriptorSetWriteType::StorageBufferDynamic(buffer_writes)
                | DescriptorSetWriteType::UniformBufferDynamic(buffer_writes)
                | DescriptorSetWriteType::StorageBuffer(buffer_writes)
                | DescriptorSetWriteType::UniformBuffer(buffer_writes) => {
                    write.descriptor_count = buffer_writes.len() as u32;
                    write.p_buffer_info = (*buffer_writes).as_ptr();
                }
                DescriptorSetWriteType::StorageTexelBuffer(buffer_views)
                | DescriptorSetWriteType::UniformTexelBuffer(buffer_views) => {
                    write.descriptor_count = buffer_views.len() as u32;
                    write.p_texel_buffer_view = (*buffer_views).as_ptr();
                }
                DescriptorSetWriteType::AccelerationStructure {
                    info,
                    accel_structs,
                } => {
                    write.descriptor_count = accel_structs.len() as u32;
                    info.p_acceleration_structures = (*accel_structs).as_ptr();
                    info.acceleration_structure_count = accel_structs.len() as u32;
                    write.p_next = info as *const _ as *const _;
                }
                DescriptorSetWriteType::InlineUniformBlock { info, data } => {
                    let data: &[u32] = *data;
                    write.descriptor_count = std::mem::size_of_val(data) as u32;
                    info.p_data = data.as_ptr() as *const _;
                    info.data_size = std::mem::size_of_val(data) as u32;
                    write.p_next = info as *const _ as *const _;
                }
            }
            write_results[i] = write;
        }
        unsafe {
            self.update_descriptor_sets(&write_results, &[]);
        }
    }
}
