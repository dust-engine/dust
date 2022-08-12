#![feature(into_future)]
#![feature(alloc_layout_extra)]

//pub mod accel_struct;
pub mod accel_struct;
pub mod camera;
pub mod geometry;
pub mod material;
pub mod pipeline;
pub mod render_asset;
pub mod renderable;
pub mod renderer;
pub mod shader;
#[cfg(feature = "swapchain")]
pub mod swapchain;
pub use ash::vk;

mod resource;

pub use resource::*;

use ash::extensions::{ext, khr};
use bevy_app::{App, AppLabel, CoreStage, Plugin};
use bevy_asset::AddAsset;
use bevy_ecs::schedule::StageLabel;
use bevy_ecs::system::Resource;
use bevy_ecs::world::World;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use crate::shader::{Shader, ShaderLoader};

#[derive(Default)]
pub struct RenderPlugin;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, AppLabel)]
pub struct RenderApp;

#[derive(Debug, Hash, PartialEq, Eq, Clone, StageLabel)]
pub enum RenderStage {
    /// Syncronize between app world and render world, write world data into GPU readable memory
    Extract,
    Prepare,
    Build,
    /// Application need to register render graph runner here.
    Render,
    Cleanup,
}

/// A "scratch" world used to avoid allocating new worlds every frame when
// swapping out the Render World.
#[derive(Default, Resource)]
struct ScratchRenderWorld(World);
impl Deref for ScratchRenderWorld {
    type Target = World;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for ScratchRenderWorld {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// The Render App World. This is only available as a resource during the Extract step.
#[derive(Default, Resource)]
pub struct RenderWorld(World);
impl Deref for RenderWorld {
    type Target = World;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for RenderWorld {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl RenderPlugin {
    fn add_render_resources(&self, render_world: &mut World) {
        let entry = Arc::new(ash::Entry::linked());
        let version = entry.try_enumerate_instance_version().unwrap().unwrap();
        let instance = dustash::Instance::create(
            entry,
            &vk::InstanceCreateInfo::builder()
                .application_info(
                    &vk::ApplicationInfo::builder()
                        .application_version(vk::make_api_version(0, 0, 1, 0))
                        .api_version(version)
                        .build(),
                )
                .enabled_extension_names(&[
                    khr::Surface::name().as_ptr(),
                    khr::Win32Surface::name().as_ptr(),
                    ext::DebugUtils::name().as_ptr(),
                ])
                .build(),
        )
        .unwrap();
        let instance = Arc::new(instance);

        let physical_devices = dustash::PhysicalDevice::enumerate(&instance).unwrap();
        let (device, queues) = physical_devices
            .into_iter()
            .next()
            .unwrap()
            .create_device(
                &[],
                &[
                    khr::Swapchain::name().as_ptr(),
                    khr::DeferredHostOperations::name().as_ptr(),
                    khr::AccelerationStructure::name().as_ptr(),
                    khr::RayTracingPipeline::name().as_ptr(),
                ],
                &vk::PhysicalDeviceFeatures2::builder()
                    .features(vk::PhysicalDeviceFeatures {
                        shader_int64: vk::TRUE,
                        shader_int16: vk::TRUE,
                        ..Default::default()
                    })
                    .push_next(&mut vk::PhysicalDeviceAccelerationStructureFeaturesKHR {
                        acceleration_structure: vk::TRUE,
                        ..Default::default()
                    })
                    .push_next(&mut vk::PhysicalDeviceRayTracingPipelineFeaturesKHR {
                        ray_tracing_pipeline: vk::TRUE,
                        ..Default::default()
                    })
                    .push_next(&mut vk::PhysicalDeviceVulkan11Features {
                        storage_buffer16_bit_access: vk::TRUE,
                        ..Default::default()
                    })
                    .push_next(&mut vk::PhysicalDeviceVulkan12Features {
                        buffer_device_address: vk::TRUE,
                        storage_buffer8_bit_access: vk::TRUE,
                        timeline_semaphore: vk::TRUE,
                        descriptor_binding_sampled_image_update_after_bind: vk::TRUE,
                        descriptor_binding_storage_image_update_after_bind: vk::TRUE,
                        descriptor_binding_uniform_buffer_update_after_bind: vk::TRUE,
                        descriptor_binding_storage_buffer_update_after_bind: vk::TRUE,
                        descriptor_binding_partially_bound: vk::TRUE,
                        runtime_descriptor_array: vk::TRUE,
                        shader_int8: vk::TRUE,
                        ..Default::default()
                    })
                    .push_next(&mut vk::PhysicalDeviceVulkan13Features {
                        synchronization2: vk::TRUE,
                        ..Default::default()
                    }),
            )
            .unwrap();
        let allocator = dustash::resources::alloc::Allocator::new(device.clone());

        let acceleration_structure_loader =
            dustash::accel_struct::AccelerationStructureLoader::new(device.clone());
        render_world.insert_resource(crate::AccelerationStructureLoader(Arc::new(
            acceleration_structure_loader,
        )));
        let ray_tracing_loader =
            dustash::ray_tracing::pipeline::RayTracingLoader::new(device.clone());
        render_world.insert_resource(crate::RayTracingLoader(Arc::new(ray_tracing_loader)));
        render_world.insert_resource(crate::Device(device));
        render_world.insert_resource(crate::Instance(instance));
        render_world.insert_resource(crate::Queues(Arc::new(queues)));
        render_world.insert_resource(crate::Allocator(Arc::new(allocator)));
    }
}
impl Plugin for RenderPlugin {
    fn build(&self, app: &mut App) {
        use bevy_ecs::schedule::{Stage, SystemStage};
        app.init_resource::<ScratchRenderWorld>()
            .add_asset::<Shader>()
            .add_asset_loader(ShaderLoader::default());

        app.register_type::<renderable::Renderable>();

        let mut render_app = App::empty();
        self.add_render_resources(&mut render_app.world);

        // Let's put certain render resources into the main world as well, since they're Arc
        app.world.insert_resource(crate::Allocator(
            render_app.world.resource::<crate::Allocator>().0.clone(),
        ));
        app.world.insert_resource(crate::Device(
            render_app.world.resource::<crate::Device>().0.clone(),
        ));

        render_app
            .add_stage(CoreStage::First, SystemStage::parallel()) // For events
            .add_stage(RenderStage::Extract, {
                let mut stage = SystemStage::parallel();
                // For the Extract stage, the system params are coming from the main world while operations wrote to Commands
                // are written to the render world. By default we automatically apply buffers after each run. We disable this
                // behavior here so that we can apply the buffer to the render world manually at a later time.
                stage.set_apply_buffers(false);
                stage
            })
            .add_stage(RenderStage::Prepare, SystemStage::parallel())
            .add_stage(
                RenderStage::Render,
                SystemStage::parallel(), //.with_system(render_system.exclusive_system().at_end()),
            )
            .add_stage(
                RenderStage::Build,
                SystemStage::parallel(), //.with_system(render_system.exclusive_system().at_end()),
            )
            .add_stage(RenderStage::Cleanup, SystemStage::parallel());

        // Add render plugins
        render_app
            .add_plugin(accel_struct::blas::BlasPlugin::default())
            .add_plugin(accel_struct::tlas::TlasPlugin::default())
            .add_plugin(swapchain::SwapchainPlugin::default())
            .add_system_to_stage(RenderStage::Extract, camera::extract_camera_system);

        // Subapp runs always get scheduled after main world runs
        app.add_sub_app(RenderApp, render_app, |app_world, render_app| {
            // reserve all existing app entities for use in render_app
            // they can only be spawned using `get_or_spawn()`
            let meta_len = app_world.entities().meta_len();
            render_app
                .world
                .entities()
                .reserve_entities(meta_len as u32);

            // flushing as "invalid" ensures that app world entities aren't added as "empty archetype" entities by default
            // these entities cannot be accessed without spawning directly onto them
            // this _only_ works as expected because clear_entities() is called at the end of every frame.
            unsafe { render_app.world.entities_mut() }.flush_as_invalid();
            {
                let render = render_app
                    .schedule
                    .get_stage_mut::<SystemStage>(&CoreStage::First)
                    .unwrap();
                render.run(&mut render_app.world);
            }
            {
                let extract = render_app
                    .schedule
                    .get_stage_mut::<SystemStage>(&RenderStage::Extract)
                    .unwrap();

                // temporarily add the render world to the app world as a resource
                let scratch_world = app_world.remove_resource::<ScratchRenderWorld>().unwrap();
                let render_world = std::mem::replace(&mut render_app.world, scratch_world.0);
                app_world.insert_resource(RenderWorld(render_world));

                extract.run(app_world);

                // add the render world back to the render app
                let render_world = app_world.remove_resource::<RenderWorld>().unwrap();
                let scratch_world = std::mem::replace(&mut render_app.world, render_world.0);
                app_world.insert_resource(ScratchRenderWorld(scratch_world));

                extract.apply_buffers(&mut render_app.world);
            }
            {
                let render = render_app
                    .schedule
                    .get_stage_mut::<SystemStage>(&RenderStage::Prepare)
                    .unwrap();
                render.run(&mut render_app.world);
            }
            {
                let render = render_app
                    .schedule
                    .get_stage_mut::<SystemStage>(&RenderStage::Build)
                    .unwrap();
                render.run(&mut render_app.world);
            }
            {
                let render = render_app
                    .schedule
                    .get_stage_mut::<SystemStage>(&RenderStage::Render)
                    .unwrap();
                render.run(&mut render_app.world);
            }
            {
                let cleanup = render_app
                    .schedule
                    .get_stage_mut::<SystemStage>(&RenderStage::Cleanup)
                    .unwrap();
                cleanup.run(&mut render_app.world);

                render_app.world.clear_entities();
            }
        });
    }

    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }
}
