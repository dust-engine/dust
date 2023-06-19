use std::sync::Arc;

use crate::Device;
use ash::{prelude::VkResult, vk};

pub struct Semaphore {
    device: Arc<Device>,
    pub(crate) semaphore: vk::Semaphore,
}

impl crate::HasDevice for Semaphore {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl crate::debug::DebugObject for Semaphore {
    const OBJECT_TYPE: vk::ObjectType = vk::ObjectType::SEMAPHORE;
    fn object_handle(&mut self) -> u64 {
        unsafe { std::mem::transmute(self.semaphore) }
    }
}

impl std::fmt::Debug for Semaphore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("Semaphore({:?})", self.semaphore))
    }
}

impl Semaphore {
    pub fn new(device: Arc<Device>) -> VkResult<Self> {
        let create_info = vk::SemaphoreCreateInfo::default();
        let semaphore = unsafe { device.create_semaphore(&create_info, None)? };
        Ok(Self { device, semaphore })
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        tracing::debug!(semaphore = ?self.semaphore, "drop semaphore");
        // Safety: Host access to semaphore must be externally synchronized
        // We have &mut self thus exclusive access to self.semaphore
        unsafe {
            self.device.destroy_semaphore(self.semaphore, None);
        }
    }
}

pub struct TimelineSemaphore {
    device: Arc<Device>,
    pub(crate) semaphore: vk::Semaphore,
}

impl crate::debug::DebugObject for TimelineSemaphore {
    const OBJECT_TYPE: vk::ObjectType = vk::ObjectType::SEMAPHORE;
    fn object_handle(&mut self) -> u64 {
        unsafe { std::mem::transmute(self.semaphore) }
    }
}

impl std::fmt::Debug for TimelineSemaphore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("TimelineSemaphore({:?})", self.semaphore))
    }
}

impl crate::HasDevice for TimelineSemaphore {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl TimelineSemaphore {
    pub fn new(device: Arc<Device>, initial_value: u64) -> VkResult<Self> {
        let type_info = vk::SemaphoreTypeCreateInfo::builder()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(initial_value)
            .build();
        let create_info = vk::SemaphoreCreateInfo {
            p_next: &type_info as *const _ as *const std::ffi::c_void,
            ..Default::default()
        };
        let semaphore = unsafe { device.create_semaphore(&create_info, None)? };
        Ok(TimelineSemaphore { device, semaphore })
    }
    pub fn signal(&self, value: u64) -> VkResult<()> {
        unsafe {
            self.device.signal_semaphore(&vk::SemaphoreSignalInfo {
                semaphore: self.semaphore,
                value,
                ..Default::default()
            })
        }
    }
    pub fn value(&self) -> VkResult<u64> {
        unsafe { self.device.get_semaphore_counter_value(self.semaphore) }
    }
    /// Block the current thread until the semaphore reaches (>=) the given value
    pub fn wait(self: &TimelineSemaphore, value: u64) -> VkResult<()> {
        unsafe {
            self.device.wait_semaphores(
                &vk::SemaphoreWaitInfo {
                    semaphore_count: 1,
                    p_semaphores: &self.semaphore,
                    p_values: &value,
                    ..Default::default()
                },
                std::u64::MAX,
            )
        }
    }
}
