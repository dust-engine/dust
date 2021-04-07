use crate::device_info::DeviceInfo;
use crate::renderer::RenderContext;
use ash::version::{DeviceV1_0, DeviceV1_1};
use ash::vk;
use dust_core::svo::alloc::{AllocError, BlockAllocator};
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

pub struct DiscreteBlock {
    system_mem: vk::DeviceMemory,
    device_mem: vk::DeviceMemory,
    offset: u64,
}

pub struct DiscreteBlockAllocator {
    block_size: u64,
    context: Arc<RenderContext>,
    bind_transfer_queue: vk::Queue,
    pub device_buffer: vk::Buffer,
    system_buffer: vk::Buffer,
    system_memtype: u32,
    device_memtype: u32,

    current_offset: u64,
    free_offsets: Vec<u64>,
    allocations: HashMap<*mut u8, DiscreteBlock>,

    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    copy_completion_fence: vk::Fence,
    sparse_binding_completion_fence: vk::Fence,
    sparse_binding_completion_semaphore: vk::Semaphore,

    // maps from resource offsets to (system_mem, device_mem)
    binds_pending: HashMap<u64, (vk::DeviceMemory, vk::DeviceMemory)>,
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
            .flags(vk::BufferCreateFlags::SPARSE_BINDING | vk::BufferCreateFlags::SPARSE_RESIDENCY);

        if graphics_queue_family == bind_transfer_queue_family {
            buffer_create_info = buffer_create_info.sharing_mode(vk::SharingMode::EXCLUSIVE);
        } else {
            buffer_create_info = buffer_create_info
                .sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&queue_family_indices);
        }
        let mut buffer_create_info = buffer_create_info
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
            .build();

        let device_buffer = device.create_buffer(&buffer_create_info, None).unwrap();
        buffer_create_info.usage = vk::BufferUsageFlags::TRANSFER_SRC;
        let system_buffer = device.create_buffer(&buffer_create_info, None).unwrap();
        let device_buf_requirements = device.get_buffer_memory_requirements(device_buffer);
        let system_buf_requirements = device.get_buffer_memory_requirements(system_buffer);
        let (system_memtype, device_memtype) = select_discrete_memtype(
            &device_info.memory_properties,
            &system_buf_requirements,
            &device_buf_requirements,
        );
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
        let sparse_binding_completion_fence = device
            .create_fence(
                &vk::FenceCreateInfo::builder()
                    .flags(vk::FenceCreateFlags::SIGNALED)
                    .build(),
                None,
            )
            .unwrap();
        let sparse_binding_completion_semaphore = device
            .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            .unwrap();
        Self {
            block_size,
            context,
            bind_transfer_queue,
            device_buffer,
            system_buffer,
            system_memtype,
            device_memtype,
            current_offset: 0,
            free_offsets: Vec::new(),
            allocations: HashMap::new(),
            command_pool,
            command_buffer,
            copy_completion_fence,
            sparse_binding_completion_fence,
            binds_pending: HashMap::new(),
            sparse_binding_completion_semaphore,
        }
    }
}

impl BlockAllocator for DiscreteBlockAllocator {
    unsafe fn allocate_block(&mut self) -> Result<*mut u8, AllocError> {
        let resource_offset = self.free_offsets.pop().unwrap_or_else(|| {
            let val = self.current_offset;
            self.current_offset += 1;
            val
        });
        let system_mem = self
            .context
            .device
            .allocate_memory(
                &vk::MemoryAllocateInfo::builder()
                    .memory_type_index(self.system_memtype)
                    .allocation_size(self.block_size)
                    .build(),
                None,
            )
            .map_err(super::utils::map_err)?;
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
        let ptr = self
            .context
            .device
            .map_memory(system_mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            .map_err(super::utils::map_err)? as *mut u8;

        if self
            .context
            .device
            .get_fence_status(self.sparse_binding_completion_fence)
            .unwrap()
        {
            // There's no active binds
            self.context
                .device
                .reset_fences(&[self.sparse_binding_completion_fence])
                .unwrap();
            // Immediately submit the request
            self.context
                .device
                .queue_bind_sparse(
                    self.bind_transfer_queue,
                    &[vk::BindSparseInfo::builder()
                        .buffer_binds(&[
                            vk::SparseBufferMemoryBindInfo::builder()
                                .buffer(self.system_buffer)
                                .binds(&[vk::SparseMemoryBind {
                                    resource_offset: resource_offset * self.block_size as u64,
                                    size: self.block_size,
                                    memory: system_mem,
                                    memory_offset: 0,
                                    flags: vk::SparseMemoryBindFlags::empty(),
                                }])
                                .build(),
                            vk::SparseBufferMemoryBindInfo::builder()
                                .buffer(self.device_buffer)
                                .binds(&[vk::SparseMemoryBind {
                                    resource_offset: resource_offset * self.block_size as u64,
                                    size: self.block_size,
                                    memory: device_mem,
                                    memory_offset: 0,
                                    flags: vk::SparseMemoryBindFlags::empty(),
                                }])
                                .build(),
                        ])
                        .build()],
                    self.sparse_binding_completion_fence,
                )
                .map_err(super::utils::map_err);
        } else {
            // Queue up the sparse binding action
            // This helps us to handle a large amount of queue binding
            // requests on application launch for example
            self.binds_pending
                .insert(resource_offset, (system_mem, device_mem));
        }
        let block = DiscreteBlock {
            system_mem,
            device_mem,
            offset: resource_offset,
        };
        self.allocations.insert(ptr, block);
        Ok(ptr)
    }

    unsafe fn deallocate_block(&mut self, block: *mut u8) {
        let block = self.allocations.remove(&block).unwrap();

        self.context.device.unmap_memory(block.system_mem);
        self.context.device.free_memory(block.system_mem, None);
        self.context.device.free_memory(block.device_mem, None);
        if self.current_offset == block.offset + 1 {
            self.current_offset -= 1;
        } else {
            self.free_offsets.push(block.offset);
        }
    }

    unsafe fn flush(&mut self, ranges: &mut dyn Iterator<Item = (*mut u8, Range<u32>)>) {
        let device = &self.context.device;
        // First, complete sparse bindings
        let needs_binding = !self.binds_pending.is_empty();
        if needs_binding {
            let len = self.binds_pending.len();
            let mut system_binds: Vec<vk::SparseMemoryBind> = Vec::with_capacity(len);
            let mut device_binds: Vec<vk::SparseMemoryBind> = Vec::with_capacity(len);
            for (resource_offsets, (system_mem, device_mem)) in self.binds_pending.drain() {
                let resource_offset = resource_offsets * self.block_size as u64;
                system_binds.push(vk::SparseMemoryBind {
                    resource_offset,
                    size: self.block_size,
                    memory: system_mem,
                    memory_offset: 0,
                    flags: vk::SparseMemoryBindFlags::empty(),
                });
                device_binds.push(vk::SparseMemoryBind {
                    resource_offset,
                    size: self.block_size,
                    memory: device_mem,
                    memory_offset: 0,
                    flags: vk::SparseMemoryBindFlags::empty(),
                });
            }
            device
                .reset_fences(&[self.sparse_binding_completion_fence])
                .unwrap();
            device
                .queue_bind_sparse(
                    self.bind_transfer_queue,
                    &[vk::BindSparseInfo::builder()
                        .signal_semaphores(&[self.sparse_binding_completion_semaphore])
                        .buffer_binds(&[
                            vk::SparseBufferMemoryBindInfo::builder()
                                .buffer(self.system_buffer)
                                .binds(&system_binds)
                                .build(),
                            vk::SparseBufferMemoryBindInfo::builder()
                                .buffer(self.device_buffer)
                                .binds(&device_binds)
                                .build(),
                        ])
                        .build()],
                    self.sparse_binding_completion_fence,
                )
                .map_err(super::utils::map_err)
                .unwrap();
        }
        let allocations = &self.allocations;
        let regions = ranges
            .map(|(block_ptr, range)| {
                let location =
                    allocations[&block_ptr].offset * self.block_size as u64 + range.start as u64;
                vk::BufferCopy {
                    src_offset: location,
                    dst_offset: location,
                    size: (range.end - range.start) as u64,
                }
            })
            .collect::<Vec<_>>();
        if regions.len() == 0 {
            return;
        }
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
        self.context.device.cmd_copy_buffer(
            self.command_buffer,
            self.system_buffer,
            self.device_buffer,
            &regions,
        );
        device.end_command_buffer(self.command_buffer).unwrap();
        device.reset_fences(&[self.copy_completion_fence]).unwrap();

        let command_buffers = [self.command_buffer];
        let submit_wait_semaphore = [self.sparse_binding_completion_semaphore];
        let submit_wait_semaphore_masks = [vk::PipelineStageFlags::TRANSFER];
        let mut submit_info = vk::SubmitInfo::builder().command_buffers(&command_buffers);
        if needs_binding {
            submit_info = submit_info
                .wait_semaphores(&submit_wait_semaphore)
                .wait_dst_stage_mask(&submit_wait_semaphore_masks);
        }
        device
            .queue_submit(
                self.bind_transfer_queue,
                &[submit_info.build()],
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
        // If there's a pending sparse binding operation in progress, also signal that
        // we're busy at the moment. New copy commands can include copy commands
        // pointing to the new memory region.
        let sparse_binding_completed = unsafe {
            self.context
                .device
                .get_fence_status(self.sparse_binding_completion_fence)
                .unwrap()
        };

        // Note that it's ok to have a copy command and a sparse binding command
        // in the queue at the same time. The copy command won't reference the newly
        // allocated memory ranges.
        copy_completed && sparse_binding_completed
    }
}

/// Returns SystemMemId, DeviceMemId
fn select_discrete_memtype(
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    system_buf_requirements: &vk::MemoryRequirements,
    device_buf_requirements: &vk::MemoryRequirements,
) -> (u32, u32) {
    let system_buf_mem_type = memory_properties.memory_types
        [0..memory_properties.memory_type_count as usize]
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
        .unwrap() as u32;

    // Search for the largest DEVICE_LOCAL heap
    let (device_heap_index, _device_heap) = memory_properties.memory_heaps
        [0..memory_properties.memory_heap_count as usize]
        .iter()
        .filter(|&heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .enumerate()
        .max_by_key(|(_, heap)| heap.size)
        .unwrap();
    let device_heap_index = device_heap_index as u32;

    let device_buf_mem_type = memory_properties.memory_types
        [0..memory_properties.memory_type_count as usize]
        .iter()
        .filter(|ty| ty.heap_index == device_heap_index)
        .enumerate()
        .position(|(id, memory_type)| {
            device_buf_requirements.memory_type_bits & (1 << id) != 0
                && memory_type
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        })
        .unwrap() as u32;

    (system_buf_mem_type, device_buf_mem_type)
}

impl Drop for DiscreteBlockAllocator {
    fn drop(&mut self) {
        let device = &self.context.device;
        unsafe {
            device.device_wait_idle().unwrap();
            device.destroy_command_pool(self.command_pool, None);
            device.destroy_buffer(self.device_buffer, None);
            device.destroy_buffer(self.system_buffer, None);
            device.destroy_fence(self.copy_completion_fence, None);
            device.destroy_fence(self.sparse_binding_completion_fence, None);
            device.destroy_semaphore(self.sparse_binding_completion_semaphore, None);
            for i in self.allocations.values() {
                device.free_memory(i.device_mem, None);
                device.free_memory(i.system_mem, None);
            }
        }
    }
}
