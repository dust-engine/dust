use crate::device_info::DeviceInfo;
use crate::renderer::RenderContext;
use ash::version::DeviceV1_0;
use ash::vk;
use crossbeam::queue::SegQueue;
use dust_core::svo::alloc::{AllocError, BlockAllocation, BlockAllocator};
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub struct DiscreteBlock {
    system_mem: vk::DeviceMemory,
    system_buf: vk::Buffer,
    device_mem: vk::DeviceMemory,
    offset: u64,
    sparse_binding_completion_semaphore: vk::Semaphore,
}

pub struct DiscreteBlockAllocator {
    block_size: u64,
    context: Arc<RenderContext>,
    bind_transfer_queue: vk::Queue,
    pub device_buffer: vk::Buffer,
    device_memtype: u32,

    current_offset: AtomicU64,
    free_offsets: SegQueue<u64>,

    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    copy_completion_fence: vk::Fence,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
}
unsafe impl Send for DiscreteBlockAllocator {}
unsafe impl Sync for DiscreteBlockAllocator {}

impl DiscreteBlockAllocator {
    pub unsafe fn new(
        context: Arc<RenderContext>,
        bind_transfer_queue: vk::Queue,
        bind_transfer_queue_family: u32,
        graphics_queue_family: u32,
        block_size: u64,
        max_storage_buffer_size: u64,
        device_info: &DeviceInfo,
    ) -> Self {
        let device = &context.device;
        let queue_family_indices = [graphics_queue_family, bind_transfer_queue_family];
        let mut buffer_create_info = vk::BufferCreateInfo::builder()
            .size(max_storage_buffer_size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
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
        let device_buf_requirements = device.get_buffer_memory_requirements(device_buffer);
        let device_memtype =
            select_device_memtype(&device_info.memory_properties, &device_buf_requirements);
        let command_pool = device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::builder()
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT)
                    .queue_family_index(bind_transfer_queue_family)
                    .build(),
                None,
            )
            .unwrap();
        let mut command_buffer = vk::CommandBuffer::null();
        device
            .fp_v1_0()
            .allocate_command_buffers(
                device.handle(),
                &vk::CommandBufferAllocateInfo::builder()
                    .command_pool(command_pool)
                    .command_buffer_count(1)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .build() as *const vk::CommandBufferAllocateInfo,
                &mut command_buffer as *mut vk::CommandBuffer,
            )
            .result()
            .unwrap();
        let copy_completion_fence = device
            .create_fence(
                &vk::FenceCreateInfo::builder()
                    .flags(vk::FenceCreateFlags::SIGNALED)
                    .build(),
                None,
            )
            .unwrap();
        Self {
            block_size,
            context,
            bind_transfer_queue,
            device_buffer,
            device_memtype,
            current_offset: AtomicU64::new(0),
            free_offsets: SegQueue::new(),
            command_pool,
            command_buffer,
            copy_completion_fence,
            memory_properties: device_info.memory_properties.clone(),
        }
    }
}

impl BlockAllocator for DiscreteBlockAllocator {
    unsafe fn allocate_block(&self) -> Result<(*mut u8, BlockAllocation), AllocError> {
        let resource_offset = self
            .free_offsets
            .pop()
            .unwrap_or_else(|| self.current_offset.fetch_add(1, Ordering::Relaxed));

        let system_buf = self
            .context
            .device
            .create_buffer(
                &vk::BufferCreateInfo::builder()
                    .size(self.block_size)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .build(),
                None,
            )
            .unwrap();

        let system_buf_requirements = self
            .context
            .device
            .get_buffer_memory_requirements(system_buf);
        let system_memtype =
            select_system_memtype(&self.memory_properties, &system_buf_requirements);
        let system_mem = self
            .context
            .device
            .allocate_memory(
                &vk::MemoryAllocateInfo::builder()
                    .memory_type_index(system_memtype)
                    .allocation_size(self.block_size)
                    .build(),
                None,
            )
            .map_err(super::utils::map_err)?;
        self.context
            .device
            .bind_buffer_memory(system_buf, system_mem, 0)
            .unwrap();
        let ptr = self
            .context
            .device
            .map_memory(system_mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            .map_err(super::utils::map_err)? as *mut u8;

        let device_mem = self
            .context
            .device
            .allocate_memory(
                &vk::MemoryAllocateInfo::builder()
                    .memory_type_index(self.device_memtype)
                    .allocation_size(self.block_size)
                    .build(),
                None,
            )
            .map_err(super::utils::map_err)?;

        let sparse_binding_completion_semaphore = self
            .context
            .device
            .create_semaphore(
                &vk::SemaphoreCreateInfo::builder().push_next(
                    &mut vk::SemaphoreTypeCreateInfo::builder()
                        .semaphore_type(vk::SemaphoreType::TIMELINE)
                        .initial_value(0)
                        .build(),
                ),
                None,
            )
            .unwrap();
        // Immediately submit the request
        self.context
            .device
            .queue_bind_sparse(
                self.bind_transfer_queue,
                &[vk::BindSparseInfo::builder()
                    .buffer_binds(&[vk::SparseBufferMemoryBindInfo::builder()
                        .buffer(self.device_buffer)
                        .binds(&[vk::SparseMemoryBind {
                            resource_offset: resource_offset * self.block_size as u64,
                            size: self.block_size,
                            memory: device_mem,
                            memory_offset: 0,
                            flags: vk::SparseMemoryBindFlags::empty(),
                        }])
                        .build()])
                    .signal_semaphores(&[sparse_binding_completion_semaphore])
                    .push_next(
                        &mut vk::TimelineSemaphoreSubmitInfo::builder()
                            .signal_semaphore_values(&[1])
                            .build(),
                    )
                    .build()],
                vk::Fence::null(),
            )
            .map_err(super::utils::map_err);
        let block = DiscreteBlock {
            system_mem,
            device_mem,
            system_buf,
            offset: resource_offset,
            sparse_binding_completion_semaphore,
        };
        let block = Box::new(block);
        let allocation = BlockAllocation(Box::into_raw(block) as u64);
        Ok((ptr, allocation))
    }

    unsafe fn deallocate_block(&self, allocation: BlockAllocation) {
        let block = allocation.0 as *mut DiscreteBlock;
        let block = Box::from_raw(block);

        self.context.device.destroy_buffer(block.system_buf, None);
        self.context.device.free_memory(block.system_mem, None);
        self.context.device.free_memory(block.device_mem, None);
        self.free_offsets.push(block.offset);
        std::mem::forget(allocation);
    }

    unsafe fn flush(&self, ranges: &mut dyn Iterator<Item = (&BlockAllocation, Range<u32>)>) {
        let device = &self.context.device;
        let mut semaphores: Vec<vk::Semaphore> = Vec::with_capacity(ranges.size_hint().0);

        self.context
            .device
            .reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty())
            .unwrap();
        self.context
            .device
            .begin_command_buffer(
                self.command_buffer,
                &vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                    .build(),
            )
            .unwrap();

        for (block_allocation, range) in ranges {
            let block = block_allocation.0 as *const DiscreteBlock;
            let block = &*block;
            let location = block.offset * self.block_size as u64 + range.start as u64;
            semaphores.push(block.sparse_binding_completion_semaphore);

            self.context.device.cmd_copy_buffer(
                self.command_buffer,
                block.system_buf,
                self.device_buffer,
                &[vk::BufferCopy {
                    src_offset: range.start as u64,
                    dst_offset: location,
                    size: (range.end - range.start) as u64,
                }],
            );
        }
        device.end_command_buffer(self.command_buffer).unwrap();
        assert!(semaphores.len() > 0);
        let wait_dst_stage_mask = vec![vk::PipelineStageFlags::TRANSFER; semaphores.len()];
        let wait_semaphore_values = vec![1; semaphores.len()];

        device.reset_fences(&[self.copy_completion_fence]).unwrap();
        let command_buffers = [self.command_buffer];
        let submit_info = vk::SubmitInfo::builder()
            .command_buffers(&command_buffers)
            .wait_semaphores(&semaphores)
            .wait_dst_stage_mask(&wait_dst_stage_mask)
            .push_next(
                &mut vk::TimelineSemaphoreSubmitInfo::builder()
                    .wait_semaphore_values(&wait_semaphore_values),
            )
            .build();
        device
            .queue_submit(
                self.bind_transfer_queue,
                &[submit_info],
                self.copy_completion_fence,
            )
            .unwrap();
    }
    fn can_flush(&self) -> bool {
        // If the previous copy hasn't completed: simply signal that we're busy at the moment.
        // The changes are going to be submitted to the queue in the next frame.
        let copy_completed = unsafe {
            self.context
                .device
                .get_fence_status(self.copy_completion_fence)
                .unwrap()
        };

        // Note that it's ok to have a copy command and a sparse binding command
        // in the queue at the same time. The copy command won't reference the newly
        // allocated memory ranges.
        copy_completed
    }
}

/// Returns SystemMemId, DeviceMemId
fn select_system_memtype(
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    system_buf_requirements: &vk::MemoryRequirements,
) -> u32 {
    memory_properties.memory_types[0..memory_properties.memory_type_count as usize]
        .iter()
        .enumerate()
        .position(|(id, memory_type)| {
            system_buf_requirements.memory_type_bits & (1 << id) != 0
                && memory_type.property_flags.contains(
                    vk::MemoryPropertyFlags::HOST_VISIBLE
                        | vk::MemoryPropertyFlags::HOST_COHERENT
                        | vk::MemoryPropertyFlags::HOST_CACHED,
                )
                && !memory_type
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        })
        .unwrap() as u32
}

fn select_device_memtype(
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    device_buf_requirements: &vk::MemoryRequirements,
) -> u32 {
    let (device_heap_index, _device_heap) = memory_properties.memory_heaps
        [0..memory_properties.memory_heap_count as usize]
        .iter()
        .filter(|&heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .enumerate()
        .max_by_key(|(_, heap)| heap.size)
        .unwrap();
    let device_heap_index = device_heap_index as u32;

    let (id, _memory_type) = memory_properties.memory_types
        [0..memory_properties.memory_type_count as usize]
        .iter()
        .enumerate()
        .filter(|(_id, ty)| ty.heap_index == device_heap_index)
        .find(|(id, memory_type)| {
            device_buf_requirements.memory_type_bits & (1 << id) != 0
                && memory_type
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        })
        .unwrap();
    id as u32
}

impl Drop for DiscreteBlockAllocator {
    fn drop(&mut self) {
        let device = &self.context.device;
        unsafe {
            device.device_wait_idle().unwrap();
            device.destroy_command_pool(self.command_pool, None);
            device.destroy_buffer(self.device_buffer, None);
            device.destroy_fence(self.copy_completion_fence, None);
        }
    }
}
