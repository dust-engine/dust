use crate::back;
use crate::hal;
use hal::prelude::*;

pub struct Frame {
    pub command_pool: <back::Backend as hal::Backend>::CommandPool,
    pub command_buffer: <back::Backend as hal::Backend>::CommandBuffer,
    pub submission_complete_fence: <back::Backend as hal::Backend>::Fence,
    pub submission_complete_semaphore: <back::Backend as hal::Backend>::Semaphore,
}

impl Frame {
    pub fn new(
        device: &<back::Backend as hal::Backend>::Device,
        graphics_queue_family: hal::queue::QueueFamilyId,
    ) -> Frame {
        let mut command_pool = unsafe {
            device
                .create_command_pool(
                    graphics_queue_family,
                    hal::pool::CommandPoolCreateFlags::TRANSIENT,
                )
                .unwrap()
        };
        let command_buffer = unsafe { command_pool.allocate_one(hal::command::Level::Primary) };
        let submission_complete_fence = device.create_fence(true).unwrap();
        let submission_complete_semaphore = device.create_semaphore().unwrap();
        Frame {
            command_pool,
            command_buffer,
            submission_complete_fence,
            submission_complete_semaphore,
        }
    }
}
