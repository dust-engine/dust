pub mod discrete;
pub mod integrated;
mod utils;

const MAX_BUFFER_SIZE: u64 = 1 << 32;

pub use discrete::DiscreteBlockAllocator;
pub use integrated::IntegratedBlockAllocator;

#[cfg(test)]
mod tests {

    use gfx_backend_vulkan as back;
    use gfx_hal as hal;
    use hal::prelude::*;

    pub(super) fn get_gpu() -> (
        back::Instance,
        hal::adapter::Gpu<back::Backend>,
        hal::adapter::MemoryProperties,
    ) {
        let instance =
            back::Instance::create("gfx_alloc_test", 1).expect("Unable to create an instance");
        let adapters = instance.enumerate_adapters();
        let adapter = {
            for adapter in &instance.enumerate_adapters() {
                println!("{:?}", adapter);
            }
            adapters
                .iter()
                .find(|adapter| adapter.info.device_type == hal::adapter::DeviceType::DiscreteGpu)
        }
        .expect("Unable to find a discrete GPU");

        let physical_device = &adapter.physical_device;
        let memory_properties = physical_device.memory_properties();
        let family = adapter
            .queue_families
            .iter()
            .find(|family| family.queue_type() == hal::queue::QueueType::Transfer)
            .expect("Can't find transfer queue family!");
        let gpu = unsafe {
            physical_device.open(
                &[(family, &[1.0])],
                hal::Features::SPARSE_BINDING | hal::Features::SPARSE_RESIDENCY_IMAGE_2D,
            )
        }
        .expect("Unable to open the physical device!");
        (instance, gpu, memory_properties)
    }
}
