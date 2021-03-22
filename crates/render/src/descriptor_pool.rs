use crate::back;
use crate::hal;
use hal::device::Device;

pub struct DescriptorPool {
    pub layout: <back::Backend as hal::Backend>::DescriptorSetLayout,
    pub pool: <back::Backend as hal::Backend>::DescriptorPool,
}

impl DescriptorPool {
    pub fn new<I>(device: &<back::Backend as hal::Backend>::Device, name: &str, bindings: I) -> Self
    where
        I: Iterator<Item = hal::pso::DescriptorSetLayoutBinding> + Clone,
    {
        let layout = unsafe {
            let mut layout = device
                .create_descriptor_set_layout(bindings.clone(), std::iter::empty())
                .unwrap();
            device.set_descriptor_set_layout_name(&mut layout, name);
            layout
        };
        let pool = unsafe {
            use hal::pso;
            device
                .create_descriptor_pool(
                    1,
                    bindings.map(|binding| pso::DescriptorRangeDesc {
                        ty: binding.ty,
                        count: binding.count,
                    }),
                    pso::DescriptorPoolCreateFlags::empty(),
                )
                .unwrap()
        };
        Self { layout, pool }
    }

    pub fn allocate_one(
        &mut self,
        device: &<back::Backend as hal::Backend>::Device,
        name: &str,
    ) -> Result<<back::Backend as hal::Backend>::DescriptorSet, hal::pso::AllocationError> {
        unsafe {
            use hal::pso::DescriptorPool;
            let mut desc_set = self.pool.allocate_one(&self.layout)?;
            device.set_descriptor_set_name(&mut desc_set, name);
            Ok(desc_set)
        }
    }

    pub fn free_one(&mut self, desc_set: <back::Backend as hal::Backend>::DescriptorSet) {
        use hal::pso::DescriptorPool;
        unsafe { self.pool.free(std::iter::once(desc_set)) }
    }
}
