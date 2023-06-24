use bevy_app::Plugin;
use rhyolite_bevy::Device;
pub use sentry;


#[cfg(feature = "aftermath")]
pub extern crate aftermath_rs as aftermath;

pub struct SentryPlugin;

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

    fn shader_debug_info(&mut self, data: &[u8]) {
    }

    fn description(&mut self, describe: &mut aftermath::DescriptionBuilder) {
    }
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

impl Plugin for SentryPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        use tracing_subscriber::prelude::*;

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
            if rhyolite::error_handler::set_global_device_lost_handler(aftermath_device_lost_handler).is_err() {
                panic!()
            }
            sentry::configure_scope(|scope| {
                scope.set_tag("nv_aftermath", true);
            });
        }

        let filter_layer = tracing_subscriber::EnvFilter::try_from_default_env()
        .or_else(|_| tracing_subscriber::EnvFilter::try_new("info"))
        .unwrap();

        // Register the Sentry tracing layer to capture breadcrumbs, events, and spans:
        tracing_subscriber::registry()
            .with(filter_layer)
            .with(tracing_subscriber::fmt::layer())
            .with(sentry_tracing::layer())
            .init();

        tracing::info!("Sentry Enabled");
        #[cfg(feature = "aftermath")]
        tracing::info!("NVIDIA Aftermath Enabled");
    }

    fn finish(&self, app: &mut bevy_app::App) {
        use rhyolite::ash::vk;
        let device: &Device = app.world.resource();

        // Query additional properties
        let mut driver_properties = vk::PhysicalDeviceDriverProperties::default();
        let mut properties2 = vk::PhysicalDeviceProperties2::builder()
            .push_next(&mut driver_properties)
            .build();
        unsafe {
            device
                .instance()
                .get_physical_device_properties2(device.physical_device().raw(), &mut properties2);
        }
        let properties = device.physical_device().properties();

        sentry::configure_scope(|scope| {
            scope.set_context(
                "gpu",
                sentry::protocol::GpuContext {
                    name: properties.device_name().to_string_lossy().into(),
                    version: Some(properties.api_version().into()),
                    driver_version: Some(properties.driver_version().into()),
                    id: Some(properties.device_id.to_string()),
                    vendor_id: Some(properties.vendor_id.to_string()),
                    vendor_name: match driver_properties.driver_id {
                        vk::DriverId::AMD_OPEN_SOURCE | vk::DriverId::AMD_PROPRIETARY | vk::DriverId::MESA_RADV => Some("AMD".into()),
                        vk::DriverId::NVIDIA_PROPRIETARY | vk::DriverId::MESA_NVK => Some("NVIDIA".into()),
                        vk::DriverId::INTEL_OPEN_SOURCE_MESA | vk::DriverId::INTEL_PROPRIETARY_WINDOWS => Some("Intel".into()),
                        vk::DriverId::ARM_PROPRIETARY => Some("Arm".into()),
                        vk::DriverId::GOOGLE_SWIFTSHADER | vk::DriverId::GGP_PROPRIETARY => Some("Google".into()),
                        vk::DriverId::MESA_LLVMPIPE => Some("Linux".into()),
                        vk::DriverId::MOLTENVK => Some("Apple".into()),
                        vk::DriverId::SAMSUNG_PROPRIETARY => Some("Samsung".into()),
                        vk::DriverId::MESA_DOZEN => Some("Microsoft".into()),
                        _ => None,
                    },
                    api_type: Some("Vulkan".to_string()),
                    other: [
                        ("driver_name".to_owned(), {
                            std::ffi::CStr::from_bytes_until_nul(unsafe {
                                std::slice::from_raw_parts(
                                    driver_properties.driver_name.as_ptr() as *const u8,
                                    driver_properties.driver_name.len(),
                                )
                            })
                            .unwrap_or_default()
                            .to_string_lossy()
                            .into()
                        }),
                        ("driver_info".to_owned(), {
                            std::ffi::CStr::from_bytes_until_nul(unsafe {
                                std::slice::from_raw_parts(
                                    driver_properties.driver_info.as_ptr() as *const u8,
                                    driver_properties.driver_info.len(),
                                )
                            })
                            .unwrap_or_default()
                            .to_string_lossy()
                            .into()
                        }),
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
                },
            )
        })
    }
}
