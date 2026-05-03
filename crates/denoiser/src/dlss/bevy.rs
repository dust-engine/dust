use std::ffi::CStr;

use super::NgxContext;

use super::sys;
use bevy::ecs::schedule::IntoScheduleConfigs;
use bevy::prelude::*;
use bevy::{app::Plugin, ecs::resource::Resource};
use bevy_pumicite::CreateDevice;
use bevy_pumicite::prelude::*;
use pumicite::device::DeviceBuilder;
use pumicite::{Instance, physical_device::PhysicalDevice, utils::AsVkHandle};

/// Bevy plugin that wires DLSS-RR (Ray Reconstruction) into the Vulkan setup.
///
/// **Plugin ordering:** must be added *after* [`bevy_pumicite::PumicitePlugin`] so
/// that the [`Instance`], [`PhysicalDevice`], and [`DeviceBuilder`] resources are
/// available. The actual Vulkan work runs as `Startup` systems before/after the
/// [`CreateDevice`] anchor.
pub struct DLSSPlugin;

impl Plugin for DLSSPlugin {
    fn build(&self, app: &mut bevy::app::App) {
        let cache_dir = app
            .world()
            .resource::<dust_app::ApplicationSettings>()
            .project_dirs
            .cache_dir();
        let path_buf = super::encode_app_data_path(cache_dir);
        let info = sys::NVSDK_NGX_FeatureDiscoveryInfo::new(&path_buf);

        let props = unsafe {
            let mut count: u32 = 0;
            let mut ptr = std::ptr::null_mut();

            sys::NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(
                &info, &mut count, &mut ptr,
            )
            .result()
            .expect("NGX: failed to fetch DLSS-RR instance extensions");
            std::slice::from_raw_parts(ptr, count as usize)
        };

        for ext in props {
            let name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
            app.add_instance_extension_named(name)
                .unwrap_or_else(|e| panic!("DLSS instance extension {name:?} unavailable: {e:?}"));
        }
        app.add_systems(
            Startup,
            (
                (|mut device_builder: ResMut<DeviceBuilder>,
                  physical_device: Res<PhysicalDevice>,
                  instance: Res<Instance>,
                  app_settings: Res<dust_app::ApplicationSettings>| {
                    let driver_properties = physical_device
                        .properties()
                        .get::<vk::PhysicalDeviceDriverProperties>();
                    if driver_properties.driver_id != vk::DriverId::NVIDIA_PROPRIETARY {
                        return;
                    }

                    let path_buf =
                        super::encode_app_data_path(app_settings.project_dirs.cache_dir());
                    let info = sys::NVSDK_NGX_FeatureDiscoveryInfo::new(&path_buf);

                    let props = unsafe {
                        let mut count: u32 = 0;
                        let mut ptr = std::ptr::null_mut();
                        sys::NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(
                            instance.handle(),
                            physical_device.vk_handle(),
                            &info,
                            &mut count,
                            &mut ptr,
                        )
                        .result()
                        .expect("NGX: failed to query DLSS-RR device extension count");
                        std::slice::from_raw_parts(ptr, count as usize)
                    };

                    for ext in props {
                        let name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
                        device_builder
                            .enable_extension_named(name)
                            .unwrap_or_else(|e| {
                                panic!("DLSS device extension {name:?} unavailable: {e:?}")
                            });
                    }
                })
                .before(CreateDevice),
                (|mut commands: Commands,
                  physical_device: Res<PhysicalDevice>,
                  app_settings: Res<dust_app::ApplicationSettings>,
                  device: Res<Device>| {
                    let driver_properties = physical_device
                        .properties()
                        .get::<vk::PhysicalDeviceDriverProperties>();
                    if driver_properties.driver_id != vk::DriverId::NVIDIA_PROPRIETARY {
                        return;
                    }

                    let path_buf =
                        super::encode_app_data_path(app_settings.project_dirs.cache_dir());

                    let ctx = NgxContext::new(device.clone(), &path_buf)
                        .expect("Unable to create NGX context");
                    commands.insert_resource(ctx);
                })
                .after(CreateDevice),
            ),
        );
    }
}

impl Resource for NgxContext {}
