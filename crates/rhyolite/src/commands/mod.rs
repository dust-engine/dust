mod pool;

pub use pool::*;

use ash::vk;

pub trait CommandBufferLike {
    fn raw_command_buffer(&self) -> vk::CommandBuffer;
}
