//! Resorce wrapper for external and Arc types

use std::{ops::Deref, sync::Arc};

use bevy_ecs::system::Resource;

#[derive(Resource)]
pub struct Device(pub Arc<dustash::Device>);
impl Deref for Device {
    type Target = Arc<dustash::Device>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl From<Arc<dustash::Device>> for Device {
    fn from(item: Arc<dustash::Device>) -> Self {
        Device(item)
    }
}

#[derive(Resource)]
pub struct Instance(pub Arc<dustash::Instance>);
impl Deref for Instance {
    type Target = Arc<dustash::Instance>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl From<Arc<dustash::Instance>> for Instance {
    fn from(item: Arc<dustash::Instance>) -> Self {
        Instance(item)
    }
}

#[derive(Resource)]
pub struct Queues(pub Arc<dustash::queue::Queues>);

impl Deref for Queues {
    type Target = Arc<dustash::queue::Queues>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl From<Arc<dustash::queue::Queues>> for Queues {
    fn from(item: Arc<dustash::queue::Queues>) -> Self {
        Queues(item)
    }
}

#[derive(Resource)]
pub struct SurfaceLoader(pub Arc<dustash::surface::SurfaceLoader>);
impl Deref for SurfaceLoader {
    type Target = Arc<dustash::surface::SurfaceLoader>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl From<Arc<dustash::surface::SurfaceLoader>> for SurfaceLoader {
    fn from(item: Arc<dustash::surface::SurfaceLoader>) -> Self {
        SurfaceLoader(item)
    }
}

#[derive(Resource)]
pub struct SwapchainLoader(pub Arc<dustash::swapchain::SwapchainLoader>);

impl Deref for SwapchainLoader {
    type Target = Arc<dustash::swapchain::SwapchainLoader>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl From<Arc<dustash::swapchain::SwapchainLoader>> for SwapchainLoader {
    fn from(item: Arc<dustash::swapchain::SwapchainLoader>) -> Self {
        SwapchainLoader(item)
    }
}

#[derive(Resource)]
pub struct RayTracingLoader(pub Arc<dustash::ray_tracing::pipeline::RayTracingLoader>);
impl Deref for RayTracingLoader {
    type Target = Arc<dustash::ray_tracing::pipeline::RayTracingLoader>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl From<Arc<dustash::ray_tracing::pipeline::RayTracingLoader>> for RayTracingLoader {
    fn from(item: Arc<dustash::ray_tracing::pipeline::RayTracingLoader>) -> Self {
        RayTracingLoader(item)
    }
}

#[derive(Resource)]
pub struct AccelerationStructureLoader(pub Arc<dustash::accel_struct::AccelerationStructureLoader>);
impl Deref for AccelerationStructureLoader {
    type Target = Arc<dustash::accel_struct::AccelerationStructureLoader>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl From<Arc<dustash::accel_struct::AccelerationStructureLoader>> for AccelerationStructureLoader {
    fn from(item: Arc<dustash::accel_struct::AccelerationStructureLoader>) -> Self {
        AccelerationStructureLoader(item)
    }
}

#[derive(Resource)]
pub struct Allocator(pub Arc<dustash::resources::alloc::Allocator>);
impl Deref for Allocator {
    type Target = Arc<dustash::resources::alloc::Allocator>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl From<Arc<dustash::resources::alloc::Allocator>> for Allocator {
    fn from(item: Arc<dustash::resources::alloc::Allocator>) -> Self {
        Allocator(item)
    }
}
