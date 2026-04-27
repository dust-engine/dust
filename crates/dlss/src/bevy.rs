use std::ffi::CStr;

use crate::NgxContext;

use super::sys;
use bevy::{app::Plugin, ecs::resource::Resource};
use bevy_pumicite::prelude::*;
use pumicite::{Instance, physical_device::PhysicalDevice, utils::AsVkHandle};

/// Bevy plugin that wires DLSS-RR (Ray Reconstruction) into the Vulkan setup.
///
/// **Plugin ordering:** must be added *before* [`bevy_pumicite::PumicitePlugin`]:
/// `build()` registers required Vulkan instance extensions before the instance
/// is created, and `finish()` registers required device extensions before the
/// device is created.
pub struct DLSSPlugin;

impl Plugin for DLSSPlugin {
    fn build(&self, app: &mut bevy::app::App) {
        let app_settings = app.world().resource::<dust_app::ApplicationSettings>();
        let instance = app.world().resource::<Instance>();
        let physical_device = app.world().resource::<PhysicalDevice>();
        let path_buf = encode_app_data_path(app_settings.project_dirs.cache_dir());
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
            app.add_device_extension_named(name)
                .unwrap_or_else(|e| panic!("DLSS device extension {name:?} unavailable: {e:?}"));
        }
    }

    fn finish(&self, app: &mut bevy::app::App) {
        let app_settings = app.world().resource::<dust_app::ApplicationSettings>();
        let device = app.world().resource::<Device>().clone();

        let path_buf = encode_app_data_path(app_settings.project_dirs.cache_dir());

        let ctx = NgxContext::new(device, &path_buf).expect("Unable to create NGX context");
        app.world_mut().insert_resource(ctx);
    }
}

impl Resource for NgxContext {}

/// A simple plugin that registers all Vulkan instance extensions used by DLSS.
pub struct DLSSInstancePlugin;
impl Plugin for DLSSInstancePlugin {
    fn build(&self, app: &mut bevy::app::App) {
        let cache_dir = app
            .world()
            .resource::<dust_app::ApplicationSettings>()
            .project_dirs
            .cache_dir();
        // NGX rejects the discovery info with FAIL_InvalidParameter if this path
        // doesn't exist, so make sure it does before encoding. Or this could be BS.
        std::fs::create_dir_all(&cache_dir).expect("failed to create DLSS cache dir");
        let path_buf = encode_app_data_path(cache_dir);
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
    }
}

/// NUL-terminated `wchar_t` encoding of `path`, suitable for
/// `NVSDK_NGX_FeatureDiscoveryInfo::ApplicationDataPath`.
///
/// The returned buffer must outlive every NGX call that reads the resulting
/// `NVSDK_NGX_FeatureDiscoveryInfo` — NGX does *not* copy the string.
fn encode_app_data_path<'a>(path: &'a std::path::Path) -> std::borrow::Cow<'a, [sys::wchar_t]> {
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        let mut v: Vec<u16> = path.as_os_str().encode_wide().collect();
        v.push(0);
        std::borrow::Cow::Owned(v)
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let bytes = cache_path.as_os_str().as_bytes();
        std::borrow::Cow::Borrowed(std::slice::from_raw_parts(
            bytes.as_ptr() as *const sys::wchar_t,
            bytes.len(),
        ))
    }
}
