use crate::Device;
use ash::{prelude::VkResult, vk};
use std::{fmt::Debug, sync::Arc};

pub struct Sampler {
    device: Arc<Device>,
    inner: vk::Sampler,
}
impl Debug for Sampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.fmt(f)
    }
}

impl Sampler {
    pub fn new(device: Arc<Device>, info: &vk::SamplerCreateInfo) -> VkResult<Self> {
        let inner = unsafe { device.create_sampler(info, None) }?;
        Ok(Self { device, inner })
    }
    pub unsafe fn raw(&self) -> vk::Sampler {
        self.inner
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe { self.device.destroy_sampler(self.inner, None) }
    }
}
