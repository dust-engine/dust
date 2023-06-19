use crate::{Device, HasDevice};
use ash::vk;

use std::sync::Arc;

/// An unsafe command pool. Command buffer lifecycles are unmanaged.
pub struct UnsafeCommandPool {
    device: Arc<Device>,
    command_pool: vk::CommandPool,
    queue_family_index: u32,
}
unsafe impl Send for UnsafeCommandPool {}
impl !Sync for UnsafeCommandPool {}

impl UnsafeCommandPool {
    pub fn new(
        device: Arc<Device>,
        queue_family_index: u32,
        flags: vk::CommandPoolCreateFlags,
    ) -> Self {
        let command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo {
                    queue_family_index,
                    flags,
                    ..Default::default()
                },
                None,
            )
        }
        .unwrap();
        Self {
            device,
            command_pool,
            queue_family_index,
        }
    }
    /// Marked unsafe because allocated command buffers won't be recycled automatically.
    pub unsafe fn allocate_n<const N: usize>(&self, secondary: bool) -> [vk::CommandBuffer; N] {
        let mut command_buffer = [vk::CommandBuffer::null(); N];
        (self.device.fp_v1_0().allocate_command_buffers)(
            self.device.handle(),
            &vk::CommandBufferAllocateInfo {
                command_pool: self.command_pool,
                level: if secondary {
                    vk::CommandBufferLevel::SECONDARY
                } else {
                    vk::CommandBufferLevel::PRIMARY
                },
                command_buffer_count: N as u32,
                ..Default::default()
            },
            command_buffer.as_mut_ptr(),
        )
        .result()
        .unwrap();
        command_buffer
    }
    pub unsafe fn free(&self, bufs: &[vk::CommandBuffer]) {
        self.device.free_command_buffers(self.command_pool, bufs)
    }
    /// Marked unsafe because allocated command buffers won't be recycled automatically.
    pub unsafe fn allocate_one(&self, secondary: bool) -> vk::CommandBuffer {
        let command_buffer: [vk::CommandBuffer; 1] = self.allocate_n(secondary);
        command_buffer[0]
    }
    pub fn reset(&mut self, release_resources: bool) {
        unsafe {
            self.device
                .reset_command_pool(
                    self.command_pool,
                    if release_resources {
                        vk::CommandPoolResetFlags::RELEASE_RESOURCES
                    } else {
                        vk::CommandPoolResetFlags::empty()
                    },
                )
                .unwrap();
        }
    }
}

impl Drop for UnsafeCommandPool {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}

pub struct SharedCommandPool {
    pool: UnsafeCommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    indice: usize,
}
impl HasDevice for SharedCommandPool {
    fn device(&self) -> &Arc<Device> {
        &self.pool.device
    }
}
impl SharedCommandPool {
    pub fn new(device: Arc<Device>, queue_family_index: u32) -> Self {
        let pool = UnsafeCommandPool::new(
            device,
            queue_family_index,
            vk::CommandPoolCreateFlags::TRANSIENT,
        );
        Self {
            pool,
            command_buffers: Vec::new(),
            indice: 0,
        }
    }
    pub fn queue_family_index(&self) -> u32 {
        self.pool.queue_family_index
    }
    pub fn allocate_one(&mut self) -> vk::CommandBuffer {
        if self.indice >= self.command_buffers.len() {
            let buffer = unsafe { self.pool.allocate_one(false) };
            self.command_buffers.push(buffer);
            self.indice += 1;
            buffer
        } else {
            let raw_buffer = self.command_buffers[self.indice];
            self.indice += 1;
            raw_buffer
        }
    }
    pub fn reset(&mut self, release_resources: bool) {
        self.indice = 0;
        self.pool.reset(release_resources);
    }
}
