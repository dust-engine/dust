//! Application-level wiring for `pumicite_super_resolution`.
//!
//! This module and the render systems in [`crate`] are the only places in dust
//! that touch super-resolution, and they do so exclusively through the generic
//! API — dust never names DLSS, MetalFX, or any concrete backend. This plugin
//! enables whatever Vulkan instance/device extensions the active backend reports
//! it needs, and publishes the application identity the renderer hands back to
//! the API on every call.

use std::ffi::CStr;
use std::path::PathBuf;

use bevy::app::{App, Plugin, Startup};
use bevy::ecs::schedule::IntoScheduleConfigs;
use bevy::prelude::*;
use bevy_pumicite::CreateDevice;
use bevy_pumicite::prelude::*;
use pumicite::device::DeviceBuilder;
use pumicite::physical_device::PhysicalDevice;
use pumicite_super_resolution::{
    SuperResolutionApplicationInfo, SuperResolutionPhysicalDevice,
    super_resolution_required_instance_extensions,
};

use dust_app::ApplicationSettings;

/// Project ID registered with the super-resolution vendor for this application
/// (for NGX, the DLSS Project ID GUID). Stable across releases so the vendor's
/// telemetry / feature gating attributes usage to dust rather than the library.
const PROJECT_ID: &CStr = c"d6922120-3e84-46b5-bf33-dabeea210fd5";
/// Application/engine version string reported alongside [`PROJECT_ID`].
const ENGINE_VERSION: &CStr = c"1.0";

/// Application identity + writable data path handed to `pumicite_super_resolution`
/// at every entry point (engine enumeration, session creation).
///
/// Inserted by [`SuperResolutionSetupPlugin`]; read by the renderer, which
/// borrows it as a [`SuperResolutionApplicationInfo`] via [`Self::info`].
#[derive(Resource, Clone)]
pub struct SuperResolutionIdentity {
    project_id: &'static CStr,
    engine_version: &'static CStr,
    application_data_path: PathBuf,
}

impl SuperResolutionIdentity {
    /// Borrows this identity as a [`SuperResolutionApplicationInfo`] for a call
    /// into the super-resolution API.
    pub fn info(&self) -> SuperResolutionApplicationInfo<'_> {
        SuperResolutionApplicationInfo {
            project_id: self.project_id,
            engine_version: self.engine_version,
            application_data_path: &self.application_data_path,
        }
    }
}

/// Backend-agnostic Vulkan setup for `pumicite_super_resolution`.
///
/// Enables the instance/device extensions the active super-resolution backend
/// requires — queried through the generic API, so dust stays unaware of the
/// concrete backend — and publishes [`SuperResolutionIdentity`]. When no backend
/// is available (e.g. a non-NVIDIA GPU) both extension queries return empty and
/// this plugin is a no-op beyond inserting the identity.
///
/// **Ordering:** add this *before* [`bevy_pumicite::PumicitePlugin`] so the
/// instance extensions are registered before the `VkInstance` is built. The
/// device-extension system runs before the [`CreateDevice`] anchor.
pub struct SuperResolutionSetupPlugin;

impl Plugin for SuperResolutionSetupPlugin {
    fn build(&self, app: &mut App) {
        let cache_dir = app
            .world()
            .resource::<ApplicationSettings>()
            .project_dirs
            .cache_dir()
            .to_path_buf();
        let identity = SuperResolutionIdentity {
            project_id: PROJECT_ID,
            engine_version: ENGINE_VERSION,
            application_data_path: cache_dir,
        };

        // Register instance extensions before the instance is created.
        super_resolution_required_instance_extensions(
            app.world_mut()
                .get_resource_or_init::<pumicite::instance::InstanceBuilder>()
                .into_inner(),
            &identity.info(),
        )
        .expect("Missing instance extensions required by super resolution engine");

        app.insert_resource(identity);

        // Enable device extensions on the builder before the logical device is
        // created. `super_resolution_required_device_extensions` gates on the
        // adapter internally, so on an unsupported GPU this loop is empty.
        app.add_systems(
            Startup,
            (|device_builder: ResMut<DeviceBuilder>,
              identity: Res<SuperResolutionIdentity>| {
                pumicite::physical_device::PhysicalDevice::super_resolution_required_device_extensions(
                    device_builder.into_inner(), &identity.info()).expect("device extensions required by super-resolution engine unavailable");
            })
            .before(CreateDevice),
        );
    }
}
