use crate::device_info::DeviceInfo;
use ash::version::DeviceV1_0;
use ash::vk;

use std::collections::HashMap;

pub struct IntegratedBlockAllocator {
    device: ash::Device,
    bind_transfer_queue: vk::Queue,
    memtype: u32,
    buffer: vk::Buffer,

    current_offset: u32,
    free_offsets: Vec<u64>,
    allocations: HashMap<*mut u8, vk::DeviceMemory>,
}

impl IntegratedBlockAllocator {
    pub unsafe fn new(
        device: ash::Device,
        bind_transfer_queue: vk::Queue,
        bind_transfer_queue_family: u32,
        graphics_queue_family: u32,
        _block_size: u64,
        max_storage_buffer_size: u64,
        device_info: &DeviceInfo,
    ) -> Self {
        let queue_family_indices = [graphics_queue_family, bind_transfer_queue_family];
        let mut buffer_create_info = vk::BufferCreateInfo::builder()
            .size(max_storage_buffer_size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .flags(vk::BufferCreateFlags::SPARSE_BINDING | vk::BufferCreateFlags::SPARSE_RESIDENCY);

        if graphics_queue_family == bind_transfer_queue_family {
            buffer_create_info = buffer_create_info.sharing_mode(vk::SharingMode::EXCLUSIVE);
        } else {
            buffer_create_info = buffer_create_info
                .sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&queue_family_indices);
        }

        let device_buffer = device
            .create_buffer(&buffer_create_info.build(), None)
            .unwrap();
        let requirements = device.get_buffer_memory_requirements(device_buffer);
        let memtype = select_integrated_memtype(&device_info.memory_properties, &requirements);
        Self {
            device,
            bind_transfer_queue,
            memtype,
            buffer: device_buffer,
            current_offset: 0,
            free_offsets: Vec::new(),
            allocations: HashMap::new(),
        }
    }
}

fn select_integrated_memtype(
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    requirements: &vk::MemoryRequirements,
) -> u32 {
    memory_properties.memory_types[0..memory_properties.memory_type_count as usize]
        .iter()
        .enumerate()
        .position(|(id, memory_type)| {
            requirements.memory_type_bits & (1 << id) != 0
                && memory_type.property_flags.contains(
                    vk::MemoryPropertyFlags::DEVICE_LOCAL
                        | vk::MemoryPropertyFlags::HOST_VISIBLE
                        | vk::MemoryPropertyFlags::HOST_COHERENT
                        | vk::MemoryPropertyFlags::HOST_CACHED,
                )
        })
        .or_else(|| {
            memory_properties.memory_types[0..memory_properties.memory_type_count as usize]
                .iter()
                .enumerate()
                .position(|(id, memory_type)| {
                    requirements.memory_type_bits & (1 << id) != 0
                        && memory_type.property_flags.contains(
                            vk::MemoryPropertyFlags::DEVICE_LOCAL
                                | vk::MemoryPropertyFlags::HOST_VISIBLE
                                | vk::MemoryPropertyFlags::HOST_COHERENT,
                        )
                })
        })
        .unwrap() as u32
}
