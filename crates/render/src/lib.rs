#![feature(let_chains)]
#![feature(generators)]

use bevy_app::{Plugin, Update};
mod blas;
mod deferred_task;
mod geometry;
mod material;
mod pipeline;
mod sbt;
mod shader;
mod tlas;
use bevy_ecs::{prelude::Component, reflect::ReflectComponent, schedule::IntoSystemConfigs};
use bevy_reflect::Reflect;
use blas::{build_blas_system, BlasStore};
pub use geometry::*;
pub use material::*;
pub use pipeline::*;
use rhyolite::ash::vk;
use rhyolite_bevy::RenderSystems;
pub use shader::*;
pub use tlas::*;

pub struct RenderPlugin {
    /// When true, the RenderPlugin will add TLASPlugin<Renderable>. As a result,
    /// a default TLASStore will be inserted into the world with all entites with a Renderable component
    /// included.
    ///
    /// In certain use cases you might want to build separate TLAS for different sets of entities. You may
    /// turn off this default behavior and add your own TLAS inclusion markers and TLASPlugin<Marker>.
    ///
    /// Default: true
    pub tlas_include_all: bool,
}
impl Default for RenderPlugin {
    fn default() -> Self {
        Self {
            tlas_include_all: true,
        }
    }
}

impl Plugin for RenderPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.add_plugin(rhyolite_bevy::RenderPlugin {
            enabled_instance_extensions: vec![
                rhyolite::ash::extensions::ext::DebugUtils::name(),
                rhyolite::ash::extensions::khr::Surface::name(),
                rhyolite::ash::extensions::khr::Win32Surface::name(),
            ],
            enabled_device_extensions: vec![
                rhyolite::ash::extensions::khr::Swapchain::name(),
                rhyolite::ash::extensions::khr::DeferredHostOperations::name(),
                rhyolite::ash::extensions::khr::AccelerationStructure::name(),
                rhyolite::ash::extensions::khr::RayTracingPipeline::name(),
            ],
            enabled_device_features: Box::new(rhyolite::PhysicalDeviceFeatures {
                v13: vk::PhysicalDeviceVulkan13Features {
                    synchronization2: vk::TRUE,
                    ..Default::default()
                },
                v12: vk::PhysicalDeviceVulkan12Features {
                    timeline_semaphore: vk::TRUE,
                    buffer_device_address: vk::TRUE,
                    ..Default::default()
                },
                acceleration_structure: vk::PhysicalDeviceAccelerationStructureFeaturesKHR {
                    acceleration_structure: vk::TRUE,
                    ..Default::default()
                },
                ray_tracing: vk::PhysicalDeviceRayTracingPipelineFeaturesKHR {
                    ray_tracing_pipeline: vk::TRUE,
                    ..Default::default()
                },
                ..Default::default()
            }),
            ..rhyolite_bevy::RenderPlugin::default()
        })
        .register_type::<Renderable>()
        .add_systems(Update, build_blas_system.in_set(RenderSystems::SetUp))
        .init_resource::<BlasStore>();

        if self.tlas_include_all {
            app.add_plugin(TLASPlugin::<Renderable>::default());
        }
    }
}

#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct Renderable {
    #[reflect(ignore)]
    pub blas_build_flags: vk::BuildAccelerationStructureFlagsKHR,
}
impl Default for Renderable {
    fn default() -> Self {
        Self {
            blas_build_flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
        }
    }
}
