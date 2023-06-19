#![feature(get_mut_unchecked)]
#![feature(generators)]

mod image;
mod loaders;
mod queue;
mod swapchain;
mod types;

use std::{
    ffi::{c_char, CStr, CString},
    sync::Arc,
};

use bevy_app::prelude::*;
use bevy_asset::AddAsset;
use bevy_ecs::prelude::*;

use rhyolite::{
    ash::{self, vk},
    cstr, Instance, Version,
};

pub use self::image::*;
pub use queue::{AsyncQueues, Frame, Queues, QueuesRouter};
pub use swapchain::{Swapchain, SwapchainConfigExt};
pub use types::*;

pub struct RenderPlugin {
    pub enabled_instance_extensions: Vec<&'static CStr>,
    pub enabled_instance_layers: Vec<&'static CStr>,

    pub enabled_device_extensions: Vec<&'static CStr>,
    pub enabled_device_features: Box<rhyolite::PhysicalDeviceFeatures>,

    pub application_name: CString,
    pub application_version: Version,
    pub engine_name: CString,
    pub engine_version: Version,
    pub api_version: Version,

    pub physical_device_index: usize,

    pub max_frame_in_flight: usize,
}
impl Default for RenderPlugin {
    fn default() -> Self {
        Self {
            application_name: cstr!(b"Unnamed Application").to_owned(),
            application_version: Default::default(),
            engine_name: cstr!(b"Unnamed Engine").to_owned(),
            engine_version: Default::default(),
            api_version: Version::new(0, 1, 3, 0),
            enabled_instance_layers: vec![],
            enabled_instance_extensions: vec![
                ash::extensions::khr::Surface::name(),
                ash::extensions::khr::Win32Surface::name(),
            ],
            physical_device_index: 0,
            max_frame_in_flight: 3,
            enabled_device_extensions: vec![ash::extensions::khr::Swapchain::name()],
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
                ..Default::default()
            }),
        }
    }
}

#[derive(Clone, Hash, Debug, PartialEq, Eq, PartialOrd, Ord, SystemSet)]
pub enum RenderSystems {
    SetUp,
    Render,
    CleanUp,
}

impl Plugin for RenderPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.configure_set(Update, RenderSystems::SetUp)
            .configure_set(Update, RenderSystems::Render.after(RenderSystems::SetUp))
            .configure_set(Update, RenderSystems::CleanUp.after(RenderSystems::Render));

        let entry = unsafe { ash::Entry::load().unwrap() };
        let instance = {
            let enabled_instance_extensions: Vec<*const c_char> = self
                .enabled_instance_extensions
                .iter()
                .map(|a| a.as_ptr())
                .collect();
            let enabled_instance_layers: Vec<*const c_char> = self
                .enabled_instance_layers
                .iter()
                .map(|a| a.as_ptr())
                .collect();
            Arc::new(
                Instance::create(
                    Arc::new(entry),
                    &rhyolite::InstanceCreateInfo {
                        enabled_extension_names: &enabled_instance_extensions,
                        enabled_layer_names: &enabled_instance_layers,
                        api_version: self.api_version.clone(),
                        engine_name: self.engine_name.as_c_str(),
                        engine_version: self.engine_version,
                        application_name: self.application_name.as_c_str(),
                        application_version: self.application_version,
                    },
                )
                .unwrap(),
            )
        };
        let physical_device = rhyolite::PhysicalDevice::enumerate(&instance)
            .unwrap()
            .into_iter()
            .skip(self.physical_device_index)
            .next()
            .unwrap();
        tracing::info!(
            "Using {:?} {:?} with memory model {:?}",
            physical_device.properties().inner.properties.device_type,
            physical_device.properties().device_name(),
            physical_device.memory_model()
        );
        let queues_router = rhyolite::QueuesRouter::new(&physical_device);

        let (device, queues) = physical_device
            .create_device(rhyolite::DeviceCreateInfo {
                enabled_features: self.enabled_device_features.clone(),
                enabled_extension_names: &self
                    .enabled_device_extensions
                    .iter()
                    .map(|a| a.as_ptr())
                    .collect::<Vec<_>>(),
                ..rhyolite::DeviceCreateInfo::with_queue_create_callback(|queue_family_index| {
                    queues_router.priorities(queue_family_index)
                })
            })
            .unwrap();
        let allocator = rhyolite::Allocator::new(device.clone());
        let device = Device::new(device);
        let queues = Queues::new(queues, self.max_frame_in_flight);

        app.insert_resource(device.clone())
            .insert_resource(queues.async_queues.clone())
            .insert_resource(queues)
            .insert_resource(QueuesRouter::new(queues_router))
            .insert_resource(Allocator::new(allocator))
            .init_resource::<StagingRingBuffer>()
            .insert_non_send_resource(swapchain::NonSendResource::default())
            .add_systems(
                Update,
                (
                    swapchain::extract_windows.in_set(RenderSystems::SetUp),
                    queue::flush_async_queue_system.in_set(RenderSystems::CleanUp),
                ),
            )
            .add_asset::<SlicedImageArray>()
            .add_asset::<Image>()
            .init_asset_loader::<loaders::PngLoader>();
    }
}
