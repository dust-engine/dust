//! DLSS Ray Reconstruction (NGX feature `RayReconstruction`, a.k.a. "DLSS-D"):
//! denoises and upscales the noisy radiance in one temporal pass.
//!
//! Only built with the `dlss` feature, which the Bazel build enables on
//! Windows and Linux; it links the NGX static library and bundles the
//! `nvngx_dlssd` runtime library that NGX loads at [`Ngx::new`].

mod sys;

use std::ffi::CStr;
use std::mem::MaybeUninit;
use std::path::{Path, PathBuf};
use std::ptr::{self, NonNull};
use std::sync::Arc;

use bevy::ecs::schedule::IntoScheduleConfigs;
use bevy::prelude::{App, Res, ResMut, Startup, UVec2};
use bevy_pumicite::CreateDevice;
use dust_app::ApplicationSettings;
use pumicite::ash::vk;
use pumicite::command::CommandEncoder;
use pumicite::device::DeviceBuilder;
use pumicite::utils::AsVkHandle;
use pumicite::Device;
use tracing::{debug, error, info, warn};

use super::{UpscalerImage, UpscalerInputs};

/// DLSS Project ID registered with NVIDIA for dust. Stable across releases so
/// NVIDIA's telemetry and feature gating attribute usage to dust.
const PROJECT_ID: &CStr = c"d6922120-3e84-46b5-bf33-dabeea210fd5";
/// Engine version string reported alongside [`PROJECT_ID`].
const ENGINE_VERSION: &CStr = c"1.0";

type NgxResult<T> = Result<T, sys::NVSDK_NGX_Result>;

impl sys::NVSDK_NGX_Result {
    fn result(self) -> NgxResult<()> {
        if self == Self::Success { Ok(()) } else { Err(self) }
    }

    /// Symbolic name from `nvsdk_ngx_defs.h`.
    fn name(self) -> &'static str {
        match self {
            Self::Success => "Success",
            Self::Fail => "Fail",
            Self::FAIL_FeatureNotSupported => "FAIL_FeatureNotSupported",
            Self::FAIL_PlatformError => "FAIL_PlatformError",
            Self::FAIL_FeatureAlreadyExists => "FAIL_FeatureAlreadyExists",
            Self::FAIL_FeatureNotFound => "FAIL_FeatureNotFound",
            Self::FAIL_InvalidParameter => "FAIL_InvalidParameter",
            Self::FAIL_ScratchBufferTooSmall => "FAIL_ScratchBufferTooSmall",
            Self::FAIL_NotInitialized => "FAIL_NotInitialized",
            Self::FAIL_UnsupportedInputFormat => "FAIL_UnsupportedInputFormat",
            Self::FAIL_RWFlagMissing => "FAIL_RWFlagMissing",
            Self::FAIL_MissingInput => "FAIL_MissingInput",
            Self::FAIL_UnableToInitializeFeature => "FAIL_UnableToInitializeFeature",
            Self::FAIL_OutOfDate => "FAIL_OutOfDate",
            Self::FAIL_OutOfGPUMemory => "FAIL_OutOfGPUMemory",
            Self::FAIL_UnsupportedFormat => "FAIL_UnsupportedFormat",
            Self::FAIL_UnableToWriteToAppDataPath => "FAIL_UnableToWriteToAppDataPath",
            Self::FAIL_UnsupportedParameter => "FAIL_UnsupportedParameter",
            Self::FAIL_Denied => "FAIL_Denied",
            Self::FAIL_NotImplemented => "FAIL_NotImplemented",
            _ => "unknown",
        }
    }
}

impl std::fmt::Debug for sys::NVSDK_NGX_Result {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({:#x})", self.name(), self.0)
    }
}

impl std::fmt::Display for sys::NVSDK_NGX_Result {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}

impl std::error::Error for sys::NVSDK_NGX_Result {}

/// Registers the Vulkan instance and device extensions NGX needs for Ray
/// Reconstruction. Called from [`super::UpscalerSetupPlugin`], before the
/// instance exists.
pub(super) fn add_vulkan_extensions(app: &mut App) {
    let app_data = encode_path(&app_data_path(
        app.world().resource::<ApplicationSettings>(),
    ));
    let discovery = discovery_info(&app_data);
    let mut count = 0;
    let mut extensions: *mut vk::ExtensionProperties = ptr::null_mut();
    // SAFETY: `discovery` and the strings it points at outlive the call. On
    // success NGX writes a pointer to a static array of `count` entries.
    let result = unsafe {
        sys::NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(
            &discovery,
            &mut count,
            &mut extensions,
        )
    };
    if result.result().is_ok() && !extensions.is_null() {
        let mut instance_builder = app
            .world_mut()
            .get_resource_or_init::<pumicite::instance::InstanceBuilder>();
        for extension in unsafe { std::slice::from_raw_parts(extensions, count as usize) } {
            let name = extension.extension_name_as_c_str().unwrap();
            if let Err(e) = instance_builder.enable_extension_named(name) {
                warn!(target: "ngx", "Instance extension {name:?} required by DLSS unavailable: {e:?}");
            }
        }
    }

    app.add_systems(
        Startup,
        enable_device_extensions.before(CreateDevice),
    );
}

fn enable_device_extensions(
    mut device_builder: ResMut<DeviceBuilder>,
    settings: Res<ApplicationSettings>,
) {
    let physical_device = device_builder.physical_device();
    if !is_nvidia_driver(physical_device) {
        return;
    }
    let app_data = encode_path(&app_data_path(&settings));
    let discovery = discovery_info(&app_data);
    let mut count = 0;
    let mut extensions: *mut vk::ExtensionProperties = ptr::null_mut();
    // SAFETY: the instance and physical device are valid, `discovery` outlives
    // the call, and on success NGX writes a pointer to a static array.
    let result = unsafe {
        sys::NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(
            physical_device.instance().handle(),
            physical_device.vk_handle(),
            &discovery,
            &mut count,
            &mut extensions,
        )
    };
    if result.result().is_err() || extensions.is_null() {
        warn!(target: "ngx", "Querying DLSS device extensions failed: {result}");
        return;
    }
    for extension in unsafe { std::slice::from_raw_parts(extensions, count as usize) } {
        let name = extension.extension_name_as_c_str().unwrap();
        if let Err(e) = device_builder.enable_extension_named(name) {
            warn!(target: "ngx", "Device extension {name:?} required by DLSS unavailable: {e:?}");
        }
    }
}

fn is_nvidia_driver(physical_device: &pumicite::physical_device::PhysicalDevice) -> bool {
    physical_device
        .properties()
        .get::<vk::PhysicalDeviceDriverProperties>()
        .driver_id
        == vk::DriverId::NVIDIA_PROPRIETARY
}

/// Writable directory NGX keeps its logs and caches in (`ApplicationDataPath`).
/// NGX requires it to exist.
fn app_data_path(settings: &ApplicationSettings) -> PathBuf {
    let path = settings.project_dirs.cache_dir().join("ngx");
    if let Err(e) = std::fs::create_dir_all(&path) {
        warn!(target: "ngx", "Could not create NGX data directory {}: {e}", path.display());
    }
    path
}

/// NUL-terminated `wchar_t` encoding of `path`, the form NGX takes paths in.
fn encode_path(path: &Path) -> Vec<sys::wchar_t> {
    #[cfg(windows)]
    let mut encoded: Vec<sys::wchar_t> = {
        use std::os::windows::ffi::OsStrExt;
        path.as_os_str().encode_wide().collect()
    };
    #[cfg(unix)]
    let mut encoded: Vec<sys::wchar_t> = {
        use std::os::unix::ffi::OsStrExt;
        path.as_os_str()
            .as_bytes()
            .iter()
            .map(|&b| b as sys::wchar_t)
            .collect()
    };
    encoded.push(0);
    encoded
}

/// Bazel runfiles path of the Ray Reconstruction runtime library. `dlss` is the
/// apparent name of the repository `MODULE.bazel` declares for the DLSS SDK.
#[cfg(target_os = "windows")]
const RUNTIME_RUNFILES_PATH: &str = "dlss/runtime/nvngx_dlssd.dll";
#[cfg(target_os = "linux")]
const RUNTIME_RUNFILES_PATH: &str = "dlss/runtime/libnvidia-ngx-dlssd.so.310.6.0";

/// Directory holding the Ray Reconstruction runtime library, which NGX loads by
/// searching the paths it is given. `DLSS_RUNTIME_DIR` overrides the bundled copy.
fn runtime_library_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("DLSS_RUNTIME_DIR").filter(|dir| !dir.is_empty()) {
        return Some(PathBuf::from(dir));
    }
    let runfiles = runfiles::Runfiles::create().ok()?;
    let source_repo = option_env!("REPOSITORY_NAME").unwrap_or("");
    let library = runfiles.rlocation_from(RUNTIME_RUNFILES_PATH, source_repo)?;
    library.parent().map(Path::to_path_buf)
}

/// Identifies dust and the Ray Reconstruction feature to NGX's feature
/// queries. The result points into `app_data`, which must outlive its uses.
fn discovery_info(app_data: &[sys::wchar_t]) -> sys::NVSDK_NGX_FeatureDiscoveryInfo {
    sys::NVSDK_NGX_FeatureDiscoveryInfo {
        SDKVersion: sys::NVSDK_NGX_VERSION_API,
        FeatureID: sys::NVSDK_NGX_Feature::RayReconstruction,
        Identifier: sys::NVSDK_NGX_Application_Identifier {
            IdentifierType: sys::NVSDK_NGX_Application_Identifier_Type::ProjectId,
            v: sys::NVSDK_NGX_Application_Identifier_Value {
                ProjectDesc: sys::NVSDK_NGX_ProjectIdDescription {
                    ProjectId: PROJECT_ID.as_ptr(),
                    EngineType: sys::NVSDK_NGX_EngineType::Custom,
                    EngineVersion: ENGINE_VERSION.as_ptr(),
                },
            },
        },
        ApplicationDataPath: app_data.as_ptr(),
        FeatureInfo: ptr::null(),
    }
}

/// Search path and logging settings for NGX. `search_dir_ptr` points at a
/// NUL-terminated directory (or is null for none) and must outlive the result.
fn feature_common_info(
    search_dir_ptr: &mut *const sys::wchar_t,
    logging: sys::NVSDK_NGX_LoggingInfo,
) -> sys::NVSDK_NGX_FeatureCommonInfo {
    sys::NVSDK_NGX_FeatureCommonInfo {
        PathListInfo: if search_dir_ptr.is_null() {
            sys::NVSDK_NGX_PathListInfo::default()
        } else {
            sys::NVSDK_NGX_PathListInfo {
                Path: search_dir_ptr,
                Length: 1,
            }
        },
        InternalData: ptr::null_mut(),
        LoggingInfo: logging,
    }
}

/// The initialized NGX runtime for dust's logical device.
///
/// NGX is a per-process singleton. `NVSDK_NGX_VULKAN_Shutdown` runs when the
/// last clone of this and the last [`RayReconstruction`] are dropped.
#[derive(Clone)]
pub struct Ngx(Arc<NgxRuntime>);

struct NgxRuntime {
    device: Device,
}

impl Drop for NgxRuntime {
    fn drop(&mut self) {
        if let Err(e) = unsafe { sys::NVSDK_NGX_VULKAN_Shutdown(self.device.vk_handle()) }.result()
        {
            warn!(target: "ngx", "NGX shutdown failed: {e}");
        }
    }
}

impl Ngx {
    /// Initializes NGX if Ray Reconstruction is supported on `device`: an
    /// NVIDIA driver, with NGX's pre-init feature query reporting support.
    pub(crate) fn new(device: &Device, settings: &ApplicationSettings) -> Result<Self, String> {
        let physical_device = device.physical_device();
        if !is_nvidia_driver(physical_device) {
            return Err("not running on the NVIDIA proprietary driver".into());
        }

        let app_data = encode_path(&app_data_path(settings));
        let search_dir = runtime_library_dir().map(|dir| encode_path(&dir));
        let mut search_dir_ptr = search_dir
            .as_ref()
            .map_or(ptr::null(), |dir| dir.as_ptr());

        let quiet = sys::NVSDK_NGX_LoggingInfo {
            LoggingCallback: None,
            MinimumLoggingLevel: sys::NVSDK_NGX_Logging_Level::Off,
            DisableOtherLoggingSinks: true,
        };
        let common_info = feature_common_info(&mut search_dir_ptr, quiet);
        let mut discovery = discovery_info(&app_data);
        discovery.FeatureInfo = &common_info;
        let mut requirement = MaybeUninit::<sys::NVSDK_NGX_FeatureRequirement>::uninit();
        // SAFETY: the instance and physical device are valid; `discovery` and
        // everything it points at outlive the call; NGX writes `requirement`
        // on success.
        unsafe {
            sys::NVSDK_NGX_VULKAN_GetFeatureRequirements(
                physical_device.instance().handle(),
                physical_device.vk_handle(),
                &discovery,
                requirement.as_mut_ptr(),
            )
        }
        .result()
        .map_err(|e| format!("feature query failed: {e}"))?;
        let requirement = unsafe { requirement.assume_init() };
        if requirement.FeatureSupported != sys::NVSDK_NGX_FeatureSupportResult_Supported {
            return Err(format!(
                "not supported (reason bits {:#x})",
                requirement.FeatureSupported
            ));
        }

        let logging = sys::NVSDK_NGX_LoggingInfo {
            LoggingCallback: Some(log_callback),
            MinimumLoggingLevel: sys::NVSDK_NGX_Logging_Level::On,
            DisableOtherLoggingSinks: true,
        };
        let common_info = feature_common_info(&mut search_dir_ptr, logging);
        // SAFETY: the handles and function pointers come from the live device;
        // the strings and `common_info` outlive the call.
        unsafe {
            sys::NVSDK_NGX_VULKAN_Init_with_ProjectID(
                PROJECT_ID.as_ptr(),
                sys::NVSDK_NGX_EngineType::Custom,
                ENGINE_VERSION.as_ptr(),
                app_data.as_ptr(),
                device.instance().handle(),
                physical_device.vk_handle(),
                device.vk_handle(),
                Some(device.instance().entry().static_fn().get_instance_proc_addr),
                Some(device.instance().fp_v1_0().get_device_proc_addr),
                &common_info,
                sys::NVSDK_NGX_VERSION_API,
            )
        }
        .result()
        .map_err(|e| format!("NGX initialization failed: {e}"))?;

        Ok(Self(Arc::new(NgxRuntime {
            device: device.clone(),
        })))
    }

    /// Creates a Ray Reconstruction feature that upscales `render_extent`
    /// inputs to `display_extent`. NGX records the feature's setup work into
    /// `encoder`.
    pub(crate) fn create_ray_reconstruction(
        &self,
        encoder: &mut CommandEncoder,
        render_extent: UVec2,
        display_extent: UVec2,
    ) -> Result<RayReconstruction, sys::NVSDK_NGX_Result> {
        use sys::NVSDK_NGX_DLSS_Feature_Flags as Flag;
        let create_params = sys::NVSDK_NGX_DLSSD_Create_Params {
            InDenoiseMode: sys::NVSDK_NGX_DLSS_Denoise_Mode::DLUnified,
            // Roughness lives in `normal.w` (see `HdrRenderTargetViews::normal`).
            InRoughnessMode: sys::NVSDK_NGX_DLSS_Roughness_Mode::Packed,
            InUseHWDepth: sys::NVSDK_NGX_DLSS_Depth_Type::HW,
            InWidth: render_extent.x,
            InHeight: render_extent.y,
            InTargetWidth: display_extent.x,
            InTargetHeight: display_extent.y,
            InPerfQualityValue: perf_quality(render_extent, display_extent),
            // HDR radiance, motion vectors at render resolution, and
            // reverse-Z depth (`near / linearViewZ`: 1 = near, 0 = far).
            InFeatureCreateFlags: Flag::IsHDR | Flag::MVLowRes | Flag::DepthInverted,
            InEnableOutputSubrects: false,
        };

        let params = ParameterMap::new()?;
        let mut handle: *mut sys::NVSDK_NGX_Handle = ptr::null_mut();
        // SAFETY: the device is the one NGX was initialized with and the
        // command buffer is recording.
        unsafe {
            sys::dust_ngx_vulkan_create_dlssd_ext1(
                self.0.device.vk_handle(),
                encoder.buffer().vk_handle(),
                1,
                1,
                &mut handle,
                params.0.as_ptr(),
                &create_params,
            )
        }
        .result()?;
        Ok(RayReconstruction {
            handle: NonNull::new(handle).expect("NGX returned a null feature handle"),
            _ngx: self.clone(),
        })
    }
}

/// Maps the upscale ratio to the NGX quality mode with the same ratio
/// (see `UpscalerQualityMode::scaling_factor`).
fn perf_quality(render_extent: UVec2, display_extent: UVec2) -> sys::NVSDK_NGX_PerfQuality_Value {
    use sys::NVSDK_NGX_PerfQuality_Value as Q;
    let ratio = display_extent.x as f32 / render_extent.x.max(1) as f32;
    if ratio < 1.15 {
        Q::DLAA
    } else if ratio < 1.4 {
        Q::UltraQuality
    } else if ratio < 1.6 {
        Q::MaxQuality
    } else if ratio < 1.85 {
        Q::Balanced
    } else if ratio < 2.5 {
        Q::MaxPerf
    } else {
        Q::UltraPerformance
    }
}

/// An empty NGX parameter map. The create/evaluate helpers fill it from their
/// typed parameter structs.
struct ParameterMap(NonNull<sys::NVSDK_NGX_Parameter>);

impl ParameterMap {
    fn new() -> NgxResult<Self> {
        let mut parameters = ptr::null_mut();
        unsafe { sys::NVSDK_NGX_VULKAN_AllocateParameters(&mut parameters) }.result()?;
        Ok(Self(NonNull::new(parameters).unwrap()))
    }
}

impl Drop for ParameterMap {
    fn drop(&mut self) {
        if let Err(e) = unsafe { sys::NVSDK_NGX_VULKAN_DestroyParameters(self.0.as_ptr()) }.result()
        {
            warn!(target: "ngx", "Destroying an NGX parameter map failed: {e}");
        }
    }
}

/// A Ray Reconstruction feature for one pair of render and display
/// resolutions. Releases the NGX feature on drop.
pub(crate) struct RayReconstruction {
    handle: NonNull<sys::NVSDK_NGX_Handle>,
    // Dropped after `handle` is released, so NGX shuts down last.
    _ngx: Ngx,
}

// SAFETY: NGX feature handles are not tied to a thread; dust drives NGX from
// one render system at a time.
unsafe impl Send for RayReconstruction {}
unsafe impl Sync for RayReconstruction {}

impl Drop for RayReconstruction {
    fn drop(&mut self) {
        if let Err(e) = unsafe { sys::NVSDK_NGX_VULKAN_ReleaseFeature(self.handle.as_ptr()) }.result()
        {
            warn!(target: "ngx", "Releasing the DLSS feature failed: {e}");
        }
    }
}

impl RayReconstruction {
    /// Records the denoise + upscale of `inputs.color` into `inputs.output`.
    /// Every image must already be in `GENERAL` layout.
    pub(crate) fn dispatch(&self, encoder: &mut CommandEncoder, inputs: &UpscalerInputs) {
        let mut color = resource(&inputs.color, false);
        let mut output = resource(&inputs.output, true);
        let mut depth = resource(&inputs.depth, false);
        let mut motion_vectors = resource(&inputs.motion_vectors, false);
        let mut diffuse_albedo = resource(&inputs.diffuse_albedo, false);
        let mut specular_albedo = resource(&inputs.specular_albedo, false);
        let mut normal = resource(&inputs.normal, false);
        let mut exposure = resource(&inputs.exposure, false);
        let mut world_to_view = inputs.world_to_view.to_cols_array();
        let mut view_to_clip = inputs.view_to_clip.to_cols_array();

        let mut eval = sys::NVSDK_NGX_VK_DLSSD_Eval_Params::zeroed();
        eval.pInColor = &mut color;
        eval.pInOutput = &mut output;
        eval.pInDepth = &mut depth;
        eval.pInMotionVectors = &mut motion_vectors;
        eval.pInDiffuseAlbedo = &mut diffuse_albedo;
        eval.pInSpecularAlbedo = &mut specular_albedo;
        // Roughness is read from `normal.w` (packed roughness mode).
        eval.pInNormals = &mut normal;
        // The color is raw radiance; the 1×1 exposure image holds the scale
        // the tonemap pass applies to it.
        eval.pInExposureTexture = &mut exposure;
        eval.InPreExposure = 1.0;
        eval.InExposureScale = 1.0;
        eval.pInWorldToViewMatrix = world_to_view.as_mut_ptr();
        eval.pInViewToClipMatrix = view_to_clip.as_mut_ptr();
        eval.InRenderSubrectDimensions = sys::NVSDK_NGX_Dimensions {
            Width: inputs.color.extent.x,
            Height: inputs.color.extent.y,
        };
        eval.InMVScaleX = 1.0;
        eval.InMVScaleY = 1.0;
        eval.InJitterOffsetX = inputs.jitter.x;
        eval.InJitterOffsetY = inputs.jitter.y;

        let params = match ParameterMap::new() {
            Ok(params) => params,
            Err(e) => {
                error!(target: "ngx", "DLSS evaluate skipped: {e}");
                return;
            }
        };
        // SAFETY: the feature is live and the command buffer is recording. NGX
        // reads the image handles inside the call; the caller keeps the images
        // locked to this command buffer until the GPU has finished with it.
        let result = unsafe {
            sys::dust_ngx_vulkan_evaluate_dlssd_ext(
                encoder.buffer().vk_handle(),
                self.handle.as_ptr(),
                params.0.as_ptr(),
                &mut eval,
            )
        };
        if let Err(e) = result.result() {
            error!(target: "ngx", "DLSS evaluate failed: {e}");
        }
    }
}

/// Describes `image` to NGX. `read_write` must be set on images NGX writes.
fn resource(image: &UpscalerImage, read_write: bool) -> sys::NVSDK_NGX_Resource_VK {
    sys::NVSDK_NGX_Resource_VK {
        Resource: sys::NVSDK_NGX_Resource_VK_Resource {
            ImageViewInfo: sys::NVSDK_NGX_ImageViewInfo_VK {
                ImageView: image.view,
                Image: image.image,
                SubresourceRange: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                Format: image.format,
                Width: image.extent.x,
                Height: image.extent.y,
            },
        },
        Type: sys::NVSDK_NGX_Resource_VK_Type::ImageView,
        ReadWrite: read_write,
    }
}

unsafe extern "C" fn log_callback(
    message: *const core::ffi::c_char,
    level: sys::NVSDK_NGX_Logging_Level,
    component: sys::NVSDK_NGX_Feature,
) {
    if message.is_null() {
        return;
    }
    let message = unsafe { CStr::from_ptr(message) }.to_string_lossy();
    let message = message.trim_end_matches(['\r', '\n']);
    match level {
        sys::NVSDK_NGX_Logging_Level::Off => {}
        sys::NVSDK_NGX_Logging_Level::On => info!(target: "ngx", ?component, "{message}"),
        sys::NVSDK_NGX_Logging_Level::Verbose => debug!(target: "ngx", ?component, "{message}"),
    }
}
