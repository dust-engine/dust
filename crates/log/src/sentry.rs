use std::{borrow::Cow, ffi::CStr};

use bevy::{app::Plugin, ecs::system::Res, log::BoxedSubscriber};
use rhyolite::{
    ash::{khr::driver_properties, vk},
    debug::{DebugUtilsMessenger, DebugUtilsMessengerCallbackData},
    Device,
};
use sentry::protocol::Map;
use tracing::{instrument::WithSubscriber, Subscriber};
use tracing_subscriber::registry::LookupSpan;

#[cfg(feature = "aftermath")]
pub extern crate aftermath_rs as aftermath;

static SENTRY_GUARD: std::sync::OnceLock<sentry::ClientInitGuard> = std::sync::OnceLock::new();

#[cfg(feature = "aftermath")]
static AFTERMATH_GUARD: std::sync::OnceLock<aftermath::Aftermath> = std::sync::OnceLock::new();

#[cfg(feature = "aftermath")]
struct AftermathDelegate;

#[cfg(feature = "aftermath")]
impl aftermath::AftermathDelegate for AftermathDelegate {
    fn dumped(&mut self, dump_data: &[u8]) {
        sentry::configure_scope(|scope| {
            scope.add_attachment(sentry::protocol::Attachment {
                buffer: dump_data.into(),
                filename: "dust.nv-gpudmp".to_owned(),
                content_type: None,
                ty: Some(sentry::protocol::AttachmentType::Attachment),
            });
        });
        sentry::capture_message("Device Lost", sentry::Level::Fatal);
    }

    fn shader_debug_info(&mut self, data: &[u8]) {}

    fn description(&mut self, describe: &mut aftermath::DescriptionBuilder) {}
}

#[cfg(feature = "aftermath")]
fn aftermath_device_lost_handler() {
    tracing::info!("Device Lost Detected");
    let status = aftermath::Status::wait_for_status(Some(std::time::Duration::from_secs(5)));
    if status != aftermath::Status::Finished {
        panic!("Unexpected crash dump status: {:?}", status);
    }
    tracing::info!("Device Lost captured by Sentry");
    sentry::end_session_with_status(sentry::protocol::SessionStatus::Crashed);
    SENTRY_GUARD.get().unwrap().close(None);
    std::process::exit(1);
}

pub fn update_subscriber(
    subscriber: impl Subscriber + for<'a> LookupSpan<'a>,
) -> impl Subscriber + for<'a> LookupSpan<'a> {
    use tracing_subscriber::prelude::*;
    let sentry_layer = sentry::integrations::tracing::layer().event_filter(|md| {
        if md.target() == "log" {
            // Ignore tracing events converted from `log`. These will be captured separately.
            return sentry::integrations::tracing::EventFilter::Ignore;
        }
        match md.level() {
            &tracing::Level::ERROR => sentry::integrations::tracing::EventFilter::Exception,
            &tracing::Level::WARN => sentry::integrations::tracing::EventFilter::Event,
            &tracing::Level::INFO => sentry::integrations::tracing::EventFilter::Breadcrumb,
            _ => sentry::integrations::tracing::EventFilter::Ignore,
        }
    });

    subscriber.with(sentry_layer)
}

fn get_gpu_ctx(device: &Device) -> sentry::protocol::GpuContext {
    use rhyolite::ash::vk;

    let properties = device.physical_device().properties();
    let driver_properties = properties.get::<vk::PhysicalDeviceDriverProperties>();

    sentry::protocol::GpuContext {
        name: properties.device_name().to_string_lossy().into(),
        version: Some(properties.api_version().into()),
        driver_version: Some(properties.driver_version().into()),
        id: Some(properties.device_id.to_string()),
        vendor_id: Some(properties.vendor_id.to_string()),
        vendor_name: match driver_properties.driver_id {
            vk::DriverId::AMD_OPEN_SOURCE
            | vk::DriverId::AMD_PROPRIETARY
            | vk::DriverId::MESA_RADV => Some("AMD".into()),
            vk::DriverId::NVIDIA_PROPRIETARY | vk::DriverId::MESA_NVK => Some("NVIDIA".into()),
            vk::DriverId::INTEL_OPEN_SOURCE_MESA | vk::DriverId::INTEL_PROPRIETARY_WINDOWS => {
                Some("Intel".into())
            }
            vk::DriverId::ARM_PROPRIETARY => Some("Arm".into()),
            vk::DriverId::GOOGLE_SWIFTSHADER | vk::DriverId::GGP_PROPRIETARY => {
                Some("Google".into())
            }
            vk::DriverId::MESA_LLVMPIPE => Some("Linux".into()),
            vk::DriverId::MOLTENVK => Some("Apple".into()),
            vk::DriverId::SAMSUNG_PROPRIETARY => Some("Samsung".into()),
            vk::DriverId::MESA_DOZEN => Some("Microsoft".into()),
            _ => None,
        },
        api_type: Some("Vulkan".to_string()),
        other: [
            (
                "driver_name".to_owned(),
                driver_properties
                    .driver_name_as_c_str()
                    .unwrap()
                    .to_string_lossy()
                    .into(),
            ),
            (
                "driver_info".to_owned(),
                driver_properties
                    .driver_info_as_c_str()
                    .unwrap()
                    .to_string_lossy()
                    .into(),
            ),
            (
                "driver_id".to_owned(),
                format!("{:?}", driver_properties.driver_id).into(),
            ),
            (
                "conformance_version".to_owned(),
                format!(
                    "{}.{}.{}.{}",
                    driver_properties.conformance_version.major,
                    driver_properties.conformance_version.minor,
                    driver_properties.conformance_version.subminor,
                    driver_properties.conformance_version.patch
                )
                .into(),
            ),
        ]
        .into_iter()
        .collect(),
        ..Default::default()
    }
}

fn config_sentry(device: Res<Device>, messenger: Option<Res<DebugUtilsMessenger>>) {
    sentry::configure_scope(|scope| scope.set_context("gpu", get_gpu_ctx(&device)));

    if let Some(messenger) = messenger {
        messenger.add_callback(debug_callback)
    }
}

pub struct SentryPlugin;
impl Plugin for SentryPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        use sentry::IntoDsn;
        let guard = sentry::init(sentry::ClientOptions {
            dsn: "https://6840bf87aa9e47b0ad2ef893529c49b3@o4505406277943296.ingest.sentry.io/4505406288363520".into_dsn().ok().unwrap_or_default(),
            release: sentry::release_name!(),
            environment: Some("development".into()),
            traces_sample_rate: 1.0,
            ..sentry::ClientOptions::default()
        });
        if SENTRY_GUARD.set(guard).is_err() {
            panic!()
        }

        #[cfg(feature = "aftermath")]
        {
            let guard = aftermath::Aftermath::new(AftermathDelegate);
            if AFTERMATH_GUARD.set(guard).is_err() {
                panic!()
            }
            if rhyolite::error_handler::set_global_device_lost_handler(
                aftermath_device_lost_handler,
            )
            .is_err()
            {
                panic!()
            }
            sentry::configure_scope(|scope| {
                scope.set_tag("nv_aftermath", true);
            });
        }

        app.add_systems(bevy::app::Startup, config_sentry);
    }
}

fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    types: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: &DebugUtilsMessengerCallbackData,
) {
    match (severity, types) {
        (
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
            | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
            vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
        ) => {
            let mut stacktrace = sentry::integrations::backtrace::current_stacktrace();
            if let Some(stacktrace) = &mut stacktrace {
                sentry::integrations::backtrace::trim_stacktrace(stacktrace, |frame, _| {
                    if let Some(function) = &frame.function {
                        function == "rhyolite::debug::debug_utils_callback"
                    } else {
                        false
                    }
                });
            }

            sentry::capture_event(sentry::protocol::Event {
                level: match severity {
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => sentry::Level::Error,
                    vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => sentry::Level::Warning,
                    _ => unreachable!(),
                },
                message: callback_data
                    .message
                    .map(CStr::to_string_lossy)
                    .map(Cow::into_owned),
                culprit: callback_data
                    .message_id_name
                    .map(CStr::to_string_lossy)
                    .map(Cow::into_owned),
                contexts: if let Some(device) = callback_data.device {
                    std::collections::BTreeMap::from_iter(vec![(
                        "gpu".to_string(),
                        get_gpu_ctx(device).into(),
                    )])
                } else {
                    std::collections::BTreeMap::new()
                },
                stacktrace,
                ..Default::default()
            });
        }
        _ => (),
    }
}
