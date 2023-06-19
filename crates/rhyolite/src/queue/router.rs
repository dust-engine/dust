use crate::{PhysicalDevice, QueueRef};
use ash::vk;

#[derive(Clone, Copy, Debug)]
pub enum QueueType {
    Graphics = 0,
    Compute = 1,
    Transfer = 2,
    SparseBinding = 3,
}
#[derive(Clone, Copy)]
struct QueueTypeMask(u8);
impl QueueTypeMask {
    pub fn empty() -> Self {
        QueueTypeMask(0)
    }
    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }
    pub fn add(&mut self, queue_type: QueueType) {
        self.0 |= 1 << queue_type as usize;
    }
    pub fn types(&self) -> QueueTypeIterator {
        QueueTypeIterator(self.0)
    }
    pub fn priority(&self) -> f32 {
        self.types().map(|a| a.priority()).sum()
    }
}
pub struct QueueTypeIterator(u8);
impl Iterator for QueueTypeIterator {
    type Item = QueueType;
    fn next(&mut self) -> Option<Self::Item> {
        if self.0 == 0 {
            return None;
        }
        let t = self.0 & self.0.overflowing_neg().0;
        let r = self.0.trailing_zeros();
        self.0 ^= t;
        Some(unsafe { std::mem::transmute(r as u8) })
    }
}

impl QueueType {
    pub fn priority(&self) -> f32 {
        // Note: Sum must be less than 1.0 but greater than 0.0
        [
            QUEUE_PRIORITY_HIGH,
            QUEUE_PRIORITY_HIGH,
            QUEUE_PRIORITY_MID,
            QUEUE_PRIORITY_LOW,
        ][*self as usize]
    }
}

pub struct QueuesRouter {
    queue_family_to_types: Vec<QueueTypeMask>,
    queue_type_to_index: [QueueRef; 4],
    queue_type_to_family: [u32; 4],
}

const QUEUE_PRIORITY_HIGH: f32 = 0.35;
const QUEUE_PRIORITY_MID: f32 = 0.2;
const QUEUE_PRIORITY_LOW: f32 = 0.1;

impl QueuesRouter {
    pub fn of_type(&self, ty: QueueType) -> QueueRef {
        self.queue_type_to_index[ty as usize]
    }
    pub fn priorities(&self, queue_family_index: u32) -> Vec<f32> {
        let types = self.queue_family_to_types[queue_family_index as usize];
        if !types.is_empty() {
            vec![types.priority()]
        } else {
            Vec::new()
        }
    }
    pub fn new(physical_device: &PhysicalDevice) -> Self {
        let available_queue_family = physical_device.get_queue_family_properties();
        Self::find_with_queue_family_properties(&available_queue_family)
    }
    fn find_with_queue_family_properties(
        available_queue_family: &[vk::QueueFamilyProperties],
    ) -> Self {
        // Must include GRAPHICS. Prefer not COMPUTE or SPARSE_BINDING.
        let graphics_queue_family = available_queue_family
            .iter()
            .enumerate()
            .filter(|&(_i, family)| family.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .max_by_key(|&(_i, family)| {
                let mut priority: i32 = 0;
                if family.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                    priority -= 1;
                }
                if family.queue_flags.contains(vk::QueueFlags::SPARSE_BINDING) {
                    priority -= 1;
                }
                (priority, family.timestamp_valid_bits)
            })
            .unwrap()
            .0 as u32;
        // Must include COMPUTE. Prefer not GRAPHICS or SPARSE_BINDING
        let compute_queue_family = available_queue_family
            .iter()
            .enumerate()
            .filter(|&(_id, family)| family.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .max_by_key(|&(_, family)| {
                // Use first compute-capable queue family
                let mut priority: i32 = 0;
                if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                    priority -= 100;
                }
                if family.queue_flags.contains(vk::QueueFlags::SPARSE_BINDING) {
                    priority -= 1;
                }
                (priority, family.timestamp_valid_bits)
            })
            .unwrap()
            .0 as u32;
        // Prefer TRANSFER, COMPUTE, then GRAPHICS.
        let transfer_queue_family = available_queue_family
            .iter()
            .enumerate()
            .max_by_key(|&(_, family)| {
                // Use first compute-capable queue family
                let mut priority: i32 = 0;
                if family.queue_flags.contains(vk::QueueFlags::TRANSFER) {
                    priority += 100;
                }
                if family.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                    priority -= 10;
                }
                if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                    priority -= 20;
                }
                if family.queue_flags.contains(vk::QueueFlags::SPARSE_BINDING) {
                    priority -= 1;
                }
                (priority, family.timestamp_valid_bits)
            })
            .unwrap()
            .0 as u32;
        let sparse_binding_queue_family = available_queue_family
            .iter()
            .enumerate()
            .filter(|&(_id, family)| family.queue_flags.contains(vk::QueueFlags::SPARSE_BINDING))
            .max_by_key(|&(_, family)| {
                // Use first compute-capable queue family
                let mut priority: i32 = 0;
                if family.queue_flags.contains(vk::QueueFlags::TRANSFER) {
                    priority -= 1;
                }
                if family.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                    priority -= 10;
                }
                if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                    priority -= 20;
                }
                (priority, family.timestamp_valid_bits)
            })
            .unwrap()
            .0 as u32;

        let mut queue_family_to_types: Vec<QueueTypeMask> =
            vec![QueueTypeMask::empty(); available_queue_family.len()];
        queue_family_to_types[sparse_binding_queue_family as usize].add(QueueType::SparseBinding);
        queue_family_to_types[transfer_queue_family as usize].add(QueueType::Transfer);
        queue_family_to_types[compute_queue_family as usize].add(QueueType::Compute);
        queue_family_to_types[graphics_queue_family as usize].add(QueueType::Graphics);

        let queue_type_to_family: [u32; 4] = [
            graphics_queue_family,
            compute_queue_family,
            transfer_queue_family,
            sparse_binding_queue_family,
        ];

        let mut queue_type_to_index: [QueueRef; 4] = [QueueRef(u8::MAX); 4];
        for (i, ty) in queue_family_to_types
            .iter()
            .filter(|x| !x.is_empty())
            .enumerate()
        {
            for queue_type in ty.types() {
                queue_type_to_index[queue_type as usize] = QueueRef(i as u8);
            }
        }
        for i in queue_type_to_index.iter() {
            assert_ne!(i.0, u8::MAX, "All queue types should've been assigned")
        }

        Self {
            queue_family_to_types,
            queue_type_to_index,
            queue_type_to_family,
        }
    }
}
