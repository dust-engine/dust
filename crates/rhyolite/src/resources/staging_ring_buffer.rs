use std::{
    ops::DerefMut,
    sync::{Arc, Weak},
};

use crate::{
    copy_buffer,
    future::{Disposable, GPUCommandFuture, RenderData, RenderRes},
    Allocator, BufferLike, Device, HasDevice, PhysicalDeviceMemoryModel,
};
use ash::{prelude::VkResult, vk};
use macros::commands;
use std::sync::Mutex;
use vma::Alloc;

struct StagingRingBufferBlock {
    device: Arc<Device>,
    parent: Weak<StagingRingBuffer>,
    block: vma::VirtualBlock,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    ptr: *mut u8,
}

unsafe impl Send for StagingRingBufferBlock {}
unsafe impl Sync for StagingRingBufferBlock {}
impl Drop for StagingRingBufferBlock {
    fn drop(&mut self) {
        if self.buffer != vk::Buffer::null() {
            unsafe {
                self.device.destroy_buffer(self.buffer, None);
            }
        }
        if self.memory != vk::DeviceMemory::null() {
            unsafe {
                self.device.free_memory(self.memory, None);
            }
        }
    }
}
struct StagingRingBufferBlockTeleporter(Option<StagingRingBufferBlock>);
impl Drop for StagingRingBufferBlockTeleporter {
    fn drop(&mut self) {
        let block = self.0.take().unwrap();
        if let Some(parent) = block.parent.upgrade() {
            parent.available_blocks.push(block);
        }
    }
}

pub struct StagingRingBuffer {
    device: Arc<Device>,
    memory_type_index: u32,
    available_blocks: crossbeam_queue::SegQueue<StagingRingBufferBlock>,
    current: Mutex<Option<Arc<StagingRingBufferBlockTeleporter>>>,
}

pub struct StagingRingBufferSlice {
    block: Arc<StagingRingBufferBlockTeleporter>,
    allocation: vma::VirtualAllocation,
    buffer: vk::Buffer,
    offset: vk::DeviceSize,
    size: vk::DeviceSize,
    ptr: *mut u8,
}
unsafe impl Send for StagingRingBufferSlice {}
unsafe impl Sync for StagingRingBufferSlice {}
impl Drop for StagingRingBufferSlice {
    fn drop(&mut self) {
        unsafe {
            self.block
                .0
                .as_ref()
                .unwrap()
                .block
                .free(&mut self.allocation)
        }
    }
}
impl RenderData for StagingRingBufferSlice {}
impl BufferLike for StagingRingBufferSlice {
    fn raw_buffer(&self) -> vk::Buffer {
        self.buffer
    }

    fn size(&self) -> vk::DeviceSize {
        self.size
    }

    fn offset(&self) -> vk::DeviceSize {
        self.offset
    }

    fn device_address(&self) -> vk::DeviceAddress {
        panic!("StagingRingBufferSlice are not supposed to be directly addressed on the GPU as they do not declare BUFFER_DEVICE_ADDRESS on creation.")
    }

    fn as_mut_ptr(&mut self) -> Option<*mut u8> {
        Some(self.ptr)
    }
}

impl StagingRingBuffer {
    const BLOCK_SIZE: usize = 1024 * 1024; // 1MB block size.
    pub fn new(device: Arc<Device>) -> VkResult<Self> {
        if let Some((memory_type_index, _)) = device
            .physical_device()
            .memory_types()
            .iter()
            .enumerate()
            .find(|(_, memory_type)| {
                memory_type
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
                    && !memory_type
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                    && !memory_type
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::HOST_CACHED)
            })
        {
            return Ok(Self {
                device,
                memory_type_index: memory_type_index as u32,
                available_blocks: Default::default(),
                current: Mutex::new(None),
            });
        } else {
            return ash::prelude::VkResult::Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
        }
    }
    unsafe fn add_new_block(self: &Arc<Self>) -> VkResult<StagingRingBufferBlockTeleporter> {
        let block = vma::VirtualBlock::new(
            vma::VirtualBlockCreateInfo::new()
                .size(Self::BLOCK_SIZE as u64)
                .flags(vma::VirtualBlockCreateFlags::VMA_VIRTUAL_BLOCK_CREATE_LINEAR_ALGORITHM_BIT),
        )?;
        let mut block = StagingRingBufferBlock {
            device: self.device.clone(),
            parent: Arc::downgrade(self),
            block,
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            ptr: std::ptr::null_mut(),
        };
        block.buffer = self.device.create_buffer(
            &vk::BufferCreateInfo {
                flags: vk::BufferCreateFlags::empty(),
                size: Self::BLOCK_SIZE as u64,
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            },
            None,
        )?;
        // If this fails and early return, destruction of buffer is automatically handled
        // by the StagingRingBufferBlock Drop trait
        block.memory = self.device.allocate_memory(
            &vk::MemoryAllocateInfo {
                allocation_size: Self::BLOCK_SIZE as u64,
                memory_type_index: self.memory_type_index,
                ..Default::default()
            },
            None,
        )?;
        self.device
            .bind_buffer_memory(block.buffer, block.memory, 0)?;
        block.ptr = self.device.map_memory(
            block.memory,
            0,
            vk::WHOLE_SIZE,
            vk::MemoryMapFlags::default(),
        )? as *mut u8;
        Ok(StagingRingBufferBlockTeleporter(Some(block)))
    }

    pub fn allocate(self: &Arc<Self>, size: vk::DeviceSize) -> VkResult<StagingRingBufferSlice> {
        let mut current_guard = self.current.lock().unwrap();
        let current = if let Some(c) = current_guard.deref_mut() {
            c
        } else {
            *current_guard = unsafe { self.add_new_block().map(|a| Some(Arc::new(a)))? };
            current_guard.as_mut().unwrap()
        };
        let current: Arc<StagingRingBufferBlockTeleporter> = current.clone();
        drop(current_guard);

        if let Ok((allocation, offset)) = unsafe {
            current.0.as_ref().unwrap().block.allocate(vma::VirtualAllocationCreateInfo {
                size,
                alignment: 0,
                user_data: 0,
                flags: vma::VirtualAllocationCreateFlags::VMA_VIRTUAL_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT
            })
        } {
            return Ok(StagingRingBufferSlice {
                buffer: current.0.as_ref().unwrap().buffer,
                ptr: unsafe { current.0.as_ref().unwrap().ptr.add(offset as usize) },
                block: current,
                allocation,
                offset,
                size,
            });
        } else {
            // Block out-of-space. Try again.
            let mut current_guard = self.current.lock().unwrap();
            *current_guard = unsafe { self.add_new_block().map(|a| Some(Arc::new(a)))? };
            let current: Arc<StagingRingBufferBlockTeleporter> = current.clone();
            drop(current_guard);
            let (allocation, offset) = unsafe {
                current.0.as_ref().unwrap().block.allocate(vma::VirtualAllocationCreateInfo {
                size,
                alignment: 0,
                user_data: 0,
                flags: vma::VirtualAllocationCreateFlags::VMA_VIRTUAL_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT
            }).unwrap()
            };
            return Ok(StagingRingBufferSlice {
                ptr: unsafe { current.0.as_ref().unwrap().ptr.add(offset as usize) },
                buffer: current.0.as_ref().unwrap().buffer,
                block: current,
                allocation,
                offset,
                size,
            });
        }
    }

    /// Update buffer with host-side data.
    /// If the buffer is host visible and mapped, this function will directly write to it.
    /// Otherwise, we use the staging ring buffer.
    pub fn update_buffer<'a>(
        self: &'a Arc<Self>,
        buffer: &'a mut RenderRes<impl BufferLike + RenderData>,
        data: &'a [u8],
    ) -> impl GPUCommandFuture<
        Output = (),
        RetainedState: 'static + Disposable,
        RecycledState: 'static + Default,
    > + 'a {
        commands! {
            let mut staging_buffer = self.allocate(buffer.inner().size()).unwrap();
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), staging_buffer.as_mut_ptr().unwrap() as *mut u8, data.len());
            }
            let staging_buffer = RenderRes::new(staging_buffer);
            copy_buffer(&staging_buffer, buffer).await;
            retain!(staging_buffer);
        }
    }
}
