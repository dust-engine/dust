#![feature(let_chains)]
#![feature(generators)]
#![feature(associated_type_defaults)]
#![feature(alloc_layout_extra)]
#![feature(int_roundings)]
#![feature(associated_type_bounds)]
#![feature(specialization)]
#![feature(btree_extract_if)]

use bevy_app::{Plugin, Update};
mod blas;
mod deferred_task;
mod geometry;
mod material;
mod noise;
mod pipeline;
mod projection;
mod sbt;
mod shader;
mod tlas;
use bevy_asset::AddAsset;
use bevy_ecs::{prelude::Component, reflect::ReflectComponent, schedule::IntoSystemConfigs};
use bevy_reflect::Reflect;
use blas::{build_blas_system, BlasStore};
use deferred_task::DeferredTaskPool;
pub use geometry::*;
pub use material::*;
pub use noise::BlueNoise;
pub use pipeline::*;
pub use projection::*;
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

    /// Use the standard pipeline.
    pub use_standard_pipeline: bool,
}
impl Default for RenderPlugin {
    fn default() -> Self {
        Self {
            tlas_include_all: true,
            use_standard_pipeline: true,
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
                rhyolite::ash::vk::KhrPipelineLibraryFn::name(),
                rhyolite::ash::vk::ExtShaderAtomicFloatFn::name(),
            ],
            enabled_device_features: Box::new(rhyolite::PhysicalDeviceFeatures {
                v13: vk::PhysicalDeviceVulkan13Features {
                    synchronization2: vk::TRUE,
                    inline_uniform_block: vk::TRUE,
                    ..Default::default()
                },
                v12: vk::PhysicalDeviceVulkan12Features {
                    timeline_semaphore: vk::TRUE,
                    buffer_device_address: vk::TRUE,
                    shader_int8: vk::TRUE,
                    storage_buffer8_bit_access: vk::TRUE,
                    shader_float16: vk::TRUE,
                    ..Default::default()
                },
                v11: vk::PhysicalDeviceVulkan11Features {
                    storage_buffer16_bit_access: vk::TRUE,
                    ..Default::default()
                },
                inner: vk::PhysicalDeviceFeatures2 {
                    features: vk::PhysicalDeviceFeatures {
                        shader_int16: vk::TRUE,
                        ..Default::default()
                    },
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
        .add_plugin(PipelineCachePlugin::default())
        .register_type::<Renderable>()
        .add_systems(Update, build_blas_system.in_set(RenderSystems::SetUp))
        .init_resource::<BlasStore>()
        .add_asset::<ShaderModule>()
        .init_resource::<BlueNoise>()
        .init_resource::<Sunlight>();

        let device = app.world.resource::<rhyolite_bevy::Device>();
        DeferredTaskPool::init(device.inner().clone());
        app.add_asset_loader(SpirvLoader::new(device.clone()));

        if self.tlas_include_all {
            app.add_plugin(TLASPlugin::<Renderable>::default());
        }
        if self.use_standard_pipeline {
            app.add_plugin(RayTracingPipelinePlugin::<StandardPipeline>::default());
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
