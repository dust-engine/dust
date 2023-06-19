mod exec;

use ash::vk;
pub use exec::*;

mod router;
pub use compile::{CompiledQueueFuture, QueueCompileExt};
pub use router::{QueueType, QueuesRouter};
mod compile;

pub struct QueueInfo {
    /// (Queue family, index in that family) indexed by queue index
    pub queues: Vec<(u32, u32)>,

    /// Mapping from queue families to queue refs
    pub families: Vec<QueueMask>,
}
impl QueueInfo {
    pub fn new(num_queue_family: u32, queue_create_infos: &[vk::DeviceQueueCreateInfo]) -> Self {
        let mut families = vec![QueueMask::empty(); num_queue_family as usize];

        let mut queues: Vec<(u32, u32)> = Vec::new();

        for info in queue_create_infos.iter() {
            for i in 0..info.queue_count {
                families[info.queue_family_index as usize].set_queue(QueueRef(queues.len() as u8));
                queues.push((info.queue_family_index, i));
            }
        }
        Self { queues, families }
    }
}
