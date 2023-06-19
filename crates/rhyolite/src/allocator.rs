use std::sync::Arc;

use crate::{Device, HasDevice};
use ash::vk;

struct AllocatorInner {
    // This needs to be defined before device, so that it gets dropped hefore device gets dropped.
    inner: vma::Allocator,
    device: Arc<Device>,
}

#[derive(Clone)]
pub struct Allocator {
    pub(crate) inner: Arc<vma::Allocator>,
    device: Arc<Device>,
}

impl Allocator {
    pub fn inner(&self) -> &vma::Allocator {
        &self.inner
    }
    pub fn new(device: Arc<Device>) -> Self {
        let mut allocator_flags: vma::AllocatorCreateFlags = vma::AllocatorCreateFlags::empty();
        if device
            .physical_device()
            .features()
            .v12
            .buffer_device_address
            == vk::TRUE
        {
            allocator_flags |= vma::AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS;
        }

        let allocator = vma::Allocator::new(
            vma::AllocatorCreateInfo::new(
                device.instance().as_ref(),
                device.as_ref(),
                device.physical_device().raw(),
            )
            .vulkan_api_version(vk::make_api_version(0, 1, 3, 0))
            .flags(allocator_flags),
        )
        .unwrap();
        Self {
            inner: Arc::new(allocator),
            device,
        }
    }
}

impl HasDevice for Allocator {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}
