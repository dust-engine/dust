//! Safe wrapper around the NVIDIA DLSS NGX SDK (Vulkan + Ray Reconstruction).
//!
//! Stage 1 surface: just `DlssError` + a result alias. Real types
//! (`NgxContext`, `DlssRrFeature`, `Resource`) land in later stages once the
//! engine has motion vectors, jitter, and split render/output resolutions.

use pumicite::ash::vk;
use pumicite::utils::AsVkHandle;
use pumicite::Device;
use std::ffi::{c_int, c_uint, c_void, c_ulonglong};
use std::mem::MaybeUninit;
use std::ptr::NonNull;
use std::{ffi::CStr, ptr};

use crate::sys::wchar_t;
pub use bevy::*;

mod bevy;
pub mod sys;

const PROJECT_ID: &CStr = c"d6922120-3e84-46b5-bf33-dabeea210fd5";
const ENGINE_VERSION: &CStr = c"1.0";

impl sys::NVSDK_NGX_Result {
    #[inline]
    pub fn result(self) -> DlssResult<()> {
        self.result_with_success(())
    }

    #[inline]
    pub fn result_with_success<T>(self, v: T) -> DlssResult<T> {
        match self {
            Self::Success => Ok(v),
            _ => Err(self),
        }
    }

    #[inline]
    pub unsafe fn assume_init_on_success<T>(self, v: std::mem::MaybeUninit<T>) -> DlssResult<T> {
        unsafe { self.result().map(move |()| v.assume_init()) }
    }

    #[inline]
    pub unsafe fn set_vec_len_on_success<T>(self, mut v: Vec<T>, len: usize) -> DlssResult<Vec<T>> {
        self.result().map(move |()| unsafe {
            v.set_len(len);
            v
        })
    }

    /// Symbolic name from `nvsdk_ngx_defs.h`, or `None` for unrecognised codes.
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
            _ => "FAIL_Unknown",
        }
    }
}

impl core::fmt::Debug for sys::NVSDK_NGX_Result {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "NVSDK_NGX_Result::{}(0x{:08x})", self.name(), self.0)
    }
}

impl core::fmt::Display for sys::NVSDK_NGX_Result {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "NVSDK_NGX_Result::{}(0x{:08x})", self.name(), self.0)
    }
}

impl core::error::Error for sys::NVSDK_NGX_Result {}

pub type DlssResult<T> = core::result::Result<T, sys::NVSDK_NGX_Result>;

pub struct ParameterMap(NonNull<sys::NVSDK_NGX_Parameter>);

impl ParameterMap {
    /// Reads a value out of the parameter map by name.
    pub fn get_param<T: NgxParam>(&self, name: &CStr) -> DlssResult<T> {
        unsafe { T::get(self.0.as_ptr(), name.as_ptr()) }
    }

    /// Writes a value into the parameter map by name.
    pub fn set_param<T: NgxParam>(&mut self, name: &CStr, value: T) {
        unsafe { T::set(self.0.as_ptr(), name.as_ptr(), value) }
    }

    /// Stores a pointer to `resource`'s underlying [`sys::NVSDK_NGX_Resource_VK`]
    /// in the parameter map. NGX does not copy the struct — `resource` must
    /// remain valid until the command buffer that consumes it has completed.
    pub fn set_resource(&mut self, name: &CStr, resource: &Resource) {
        self.set_param(name, resource.as_ngx_ptr());
    }
}

impl Drop for ParameterMap {
    fn drop(&mut self) {
        unsafe {
            if let Err(e) = sys::NVSDK_NGX_VULKAN_DestroyParameters(self.0.as_ptr()).result() {
                tracing::warn!(target: "ngx", "DestroyParameters failed: {e}");
            }
        }
    }
}

mod sealed {
    pub trait Sealed {}
}

/// Types transferable to/from a [`ParameterMap`] via
/// [`ParameterMap::get_param`] / [`ParameterMap::set_param`].
///
/// Sealed: only the primitive types matching the NGX C API getter+setter pairs
/// implement this trait — `u64`, `f32`, `u32`, `c_int`, `*mut c_void`. NGX also
/// exposes `SetD` (`f64`) but no matching `GetD`, so `f64` is intentionally
/// excluded; if you need to write a double, cast to `f32` or write the bits
/// through `*mut c_void`.
pub trait NgxParam: sealed::Sealed + Sized {
    /// # Safety
    /// `map` must be a valid parameter-map pointer and `name` must be a
    /// NUL-terminated C string. NGX writes the output value only on success.
    unsafe fn get(
        map: *mut sys::NVSDK_NGX_Parameter,
        name: *const std::ffi::c_char,
    ) -> DlssResult<Self>;

    /// # Safety
    /// `map` must be a valid parameter-map pointer and `name` must be a
    /// NUL-terminated C string.
    unsafe fn set(
        map: *mut sys::NVSDK_NGX_Parameter,
        name: *const std::ffi::c_char,
        value: Self,
    );
}

macro_rules! impl_ngx_param {
    ($t:ty, $getter:ident, $setter:ident) => {
        impl sealed::Sealed for $t {}
        impl NgxParam for $t {
            unsafe fn get(
                map: *mut sys::NVSDK_NGX_Parameter,
                name: *const std::ffi::c_char,
            ) -> DlssResult<Self> {
                let mut out = MaybeUninit::<Self>::uninit();
                unsafe {
                    sys::$getter(map, name, out.as_mut_ptr())
                        .assume_init_on_success(out)
                }
            }

            unsafe fn set(
                map: *mut sys::NVSDK_NGX_Parameter,
                name: *const std::ffi::c_char,
                value: Self,
            ) {
                unsafe { sys::$setter(map, name, value) }
            }
        }
    };
}

impl_ngx_param!(c_ulonglong, NVSDK_NGX_Parameter_GetULL, NVSDK_NGX_Parameter_SetULL);
impl_ngx_param!(f32, NVSDK_NGX_Parameter_GetF, NVSDK_NGX_Parameter_SetF);
impl_ngx_param!(c_uint, NVSDK_NGX_Parameter_GetUI, NVSDK_NGX_Parameter_SetUI);
impl_ngx_param!(c_int, NVSDK_NGX_Parameter_GetI, NVSDK_NGX_Parameter_SetI);
impl_ngx_param!(*mut c_void, NVSDK_NGX_Parameter_GetVoidPointer, NVSDK_NGX_Parameter_SetVoidPointer);

impl sys::NVSDK_NGX_FeatureDiscoveryInfo {
    /// `app_data_path` must be a NUL-terminated `wchar_t` string and must remain
    /// alive for as long as the returned struct is used by NGX.
    pub fn new(app_data_path: &[sys::wchar_t]) -> Self {
        debug_assert!(
            app_data_path.last() == Some(&0),
            "ApplicationDataPath must be NUL-terminated"
        );
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
            ApplicationDataPath: app_data_path.as_ptr(),
            FeatureInfo: std::ptr::null(),
        }
    }
}

/// Implicitly owns the NGX context. NGX functions are non-threadsafe, so we generally
/// require the caller to NGX functions to acquire mutable ownership to this struct.
pub struct NgxContext {
    device: Device,
}

unsafe impl Send for NgxContext {}
unsafe impl Sync for NgxContext {}

impl NgxContext {
    /// Allocates the capability parameter map and probes DLSS-RR availability.
    /// Must be called only after `NVSDK_NGX_VULKAN_Init_with_ProjectID` succeeded.
    ///
    /// `feature_search_paths` are passed to NGX as `PathListInfo` and
    /// determine where the SDK looks for the `nvngx_dlssd.dll` /
    /// `libnvidia-ngx-dlssd.so` runtime in addition to the application
    /// directory. Each path must be a NUL-terminated `wchar_t` slice.
    fn new(
        device: Device,
        application_data_path: &[wchar_t],
    ) -> DlssResult<Self> {
        assert_eq!(
            application_data_path.last(),
            Some(&0),
            "application_data_path must be null-terminated"
        );

        let runfile_dll_dir = match runfiles::Runfiles::create().and_then(|runfiles| {
                #[cfg(target_os = "windows")]
                const DLSSD_RUNFILES_PATH: &str = "dlss/lib/Windows_x86_64/rel/nvngx_dlssd.dll";
                #[cfg(target_os = "linux")]
                const DLSSD_RUNFILES_PATH: &str = "dlss/lib/Linux_x86_64/rel/libnvidia-ngx-dlssd.so.310.6.0";
                runfiles::rlocation!(runfiles, DLSSD_RUNFILES_PATH).ok_or(runfiles::RunfilesError::RunfilesDirNotFound)
        }) {
            Ok(runfiles) => Some(runfiles),
            Err(e) => {
                tracing::warn!("DLSS DLLs not found in runfiles dir {}", e);
                None
            },
        };
        let runfile_dll_dir = runfile_dll_dir.as_ref().and_then(|x| x.parent()).map(encode_app_data_path);
        let mut runfile_dll_dir_ptr = runfile_dll_dir.map(|x| x.as_ptr()).unwrap_or_default();

        unsafe {
            sys::NVSDK_NGX_VULKAN_Init_with_ProjectID(
                PROJECT_ID.as_ptr(),
                sys::NVSDK_NGX_EngineType::Custom,
                ENGINE_VERSION.as_ptr(),
                application_data_path.as_ptr(),
                device.instance().handle(),
                device.physical_device().vk_handle(),
                device.vk_handle(),
                Some(device.instance().entry().static_fn().get_instance_proc_addr),
                Some(device.instance().fp_v1_0().get_device_proc_addr),
                &sys::NVSDK_NGX_FeatureCommonInfo {
                    PathListInfo: if runfile_dll_dir_ptr.is_null() {
                            sys::NVSDK_NGX_PathListInfo::default()
                        } else {
                            sys::NVSDK_NGX_PathListInfo {
                                Path: &mut runfile_dll_dir_ptr,
                                Length: 1,
                            }
                        },
                    InternalData: std::ptr::null_mut(),
                    LoggingInfo: sys::NVSDK_NGX_LoggingInfo {
                        LoggingCallback: Some(ngx_log_callback),
                        MinimumLoggingLevel: sys::NVSDK_NGX_Logging_Level::On,
                        DisableOtherLoggingSinks: true,
                    },
                },
                sys::NVSDK_NGX_VERSION_API,
            )
            .result()?;
        }
        Ok(Self { device })
    }

    /// Allocates a parameter map used to set parameters needed by the SDK.
    ///
    /// Allocates a new NVSDK_NGX_Parameter map for providing parameters to the
    // SDK. The lifetime of this parameter map must be managed by the
    // application. The NVSDK_NGX_Parameter interface allows simple parameter
    // setup using named fields. For example, set the width by calling
    // Parameters->Set(NVSDK_NGX_Parameter_Width, 100) or provide a resource
    // pointer by calling Parameters->Set(NVSDK_NGX_Parameter_Color, resource).
    // For more details, see the sample code.
    //
    // Use NVSDK_NGX_DestroyParameters to free a parameter map created by
    // NVSDK_NGX_AllocateParameters. Parameter maps created by
    // NVSDK_NGX_AllocateParameters must NOT be freed using the free/delete
    // operator.
    //
    // Parameter maps created by NVSDK_NGX_AllocateParameters do not come
    // pre-populated with NGX capabilities and available features. To create a
    // new parameter map pre-populated with such information, use
    // NVSDK_NGX_GetCapabilityParameters instead.
    //
    // This function may return NVSDK_NGX_Result_FAIL_OutOfDate if using an
    // older driver that does not support this API call. In such a case,
    // NVSDK_NGX_GetParameters may be used as a fallback.
    pub fn allocate_parameters(&mut self) -> DlssResult<ParameterMap> {
        let mut parameters: *mut sys::NVSDK_NGX_Parameter = ptr::null_mut();
        unsafe {
            sys::NVSDK_NGX_VULKAN_AllocateParameters(&mut parameters).result()?;
        }
        Ok(ParameterMap(NonNull::new(parameters).unwrap()))
    }
    // Allocates a parameter map populated with NGX and feature capabilities.
    //
    // Allocates a new NVSDK_NGX_Parameter map pre-populated with NGX
    // capabilities and information about available features. The output
    // parameter map can also be used in the same ways as a parameter map
    // allocated with NVSDK_NGX_AllocateParameters. However, it is not
    // recommended to use NVSDK_NGX_GetCapabilityParameters unless querying NGX
    // capabilities due to the overhead associated with pre-populating the
    // parameter map.
    //
    // Use NVSDK_NGX_DestroyParameters to free a parameter map created by
    // NVSDK_NGX_GetCapabilityParameters. Parameter maps created by
    // NVSDK_NGX_GetCapabilityParameters must NOT be freed using the
    // free/delete operator.
    //
    // This function may return NVSDK_NGX_Result_FAIL_OutOfDate if using an
    // older driver that does not support this API call. In such a case,
    // NVSDK_NGX_GetParameters may be used as a fallback.
    pub fn get_capability_parameters(&mut self) -> DlssResult<ParameterMap> {
        let mut parameters: *mut sys::NVSDK_NGX_Parameter = ptr::null_mut();
        unsafe {
            sys::NVSDK_NGX_VULKAN_GetCapabilityParameters(&mut parameters).result()?;
        }
        Ok(ParameterMap(NonNull::new(parameters).unwrap()))
    }

    /// Probes the driver capability map for DLSS-RR (Ray Reconstruction).
    ///
    /// Returns `Ok(())` if the runtime reports the feature as available. If
    /// unavailable, surfaces the most specific failure NGX provides — the
    /// per-feature `FeatureInitResult` if set, otherwise
    /// [`sys::NVSDK_NGX_Result::FAIL_FeatureNotSupported`]. A non-zero
    /// `NeedsUpdatedDriver` flag is logged at warn level.
    pub fn check_dlss_rr_available(&mut self) -> DlssResult<()> {
        let caps = self.get_capability_parameters()?;

        let available: c_uint = caps
            .get_param(sys::params::SuperSamplingDenoising_Available)
            .unwrap_or(0);

        if available != 0 {
            return Ok(());
        }

        if let Ok(needs_update) =
            caps.get_param::<c_uint>(sys::params::SuperSamplingDenoising_NeedsUpdatedDriver)
            && needs_update != 0
        {
            tracing::warn!(target: "ngx", "DLSS-RR unavailable: driver update required");
        }

        let init_result: c_int = caps
            .get_param(sys::params::SuperSamplingDenoising_FeatureInitResult)
            .unwrap_or(0);
        if init_result != 0 {
            return Err(sys::NVSDK_NGX_Result(init_result as u32));
        }
        Err(sys::NVSDK_NGX_Result::FAIL_FeatureNotSupported)
    }

    /// Creates a DLSS-RR (Ray Reconstruction) feature on `cmd_buffer`.
    ///
    /// Mirrors the `NGX_VULKAN_CREATE_DLSSD_EXT` C helper: checks runtime
    /// availability, allocates a parameter map populated from `create_params`,
    /// and calls `NVSDK_NGX_VULKAN_CreateFeature1` with
    /// [`sys::NVSDK_NGX_Feature::RayReconstruction`].
    ///
    /// `cmd_buffer` must be in the recording state — NGX records its own
    /// initialization commands into it. The caller is responsible for
    /// submitting the buffer before evaluating the returned feature.
    pub fn create_dlssd_feature(
        &mut self,
        cmd_buffer: pumicite::ash::vk::CommandBuffer,
        create_params: &sys::NVSDK_NGX_DLSSD_Create_Params,
    ) -> DlssResult<NgxFeature> {
        let mut params = self.allocate_parameters()?;
        let mut handle: *mut sys::NVSDK_NGX_Handle = ptr::null_mut();
        unsafe {
            sys::dust_ngx_vulkan_create_dlssd_ext1(
                self.device.vk_handle(),
                cmd_buffer,
                1,   // Multi GPU only (default 1)
                1, // Multi GPU only (default 1)
                &mut handle,
                params.0.as_ptr(),
                create_params,
            )
            .result()?;
        }

        Ok(NgxFeature {
            handle: NonNull::new(handle).expect("NGX returned null handle on success"),
        })
    }

    /// Records a DLSS-RR evaluate dispatch into `cmd_buffer`.
    ///
    /// Mirrors the `NGX_VULKAN_EVALUATE_DLSSD_EXT` C helper: writes every
    /// field of `eval_params` into a freshly-allocated parameter map and
    /// calls `NVSDK_NGX_VULKAN_EvaluateFeature_C`. NGX consumes the resource
    /// pointers stored in `eval_params` *during the recorded dispatch*, so
    /// the `NVSDK_NGX_Resource_VK` values referenced by those pointers — and
    /// the underlying images, image views, and backing memory — must remain
    /// valid until the GPU has finished executing `cmd_buffer`.
    ///
    /// `cmd_buffer` must be in the recording state and must have been
    /// allocated from a queue family that supports compute.
    pub fn evaluate_dlssd(
        &mut self,
        cmd_buffer: pumicite::ash::vk::CommandBuffer,
        feature: &NgxFeature,
        eval_params: &mut sys::NVSDK_NGX_VK_DLSSD_Eval_Params,
    ) -> DlssResult<()> {
        let params = self.allocate_parameters()?;
        unsafe {
            sys::dust_ngx_vulkan_evaluate_dlssd_ext(
                cmd_buffer,
                feature.handle.as_ptr(),
                params.0.as_ptr(),
                eval_params,
            )
            .result()
        }
    }
}

/// Owns an NGX feature handle (e.g. a DLSS-RR instance).
///
/// Created via [`NgxContext::create_dlss_rr_feature`]. Drop calls
/// `NVSDK_NGX_VULKAN_ReleaseFeature`, which must run before the
/// [`NgxContext`] that produced it is shut down.
pub struct NgxFeature {
    handle: NonNull<sys::NVSDK_NGX_Handle>,
}

unsafe impl Send for NgxFeature {}
unsafe impl Sync for NgxFeature {}

impl NgxFeature {
    /// Raw NGX handle pointer, for `NVSDK_NGX_VULKAN_EvaluateFeature_C`.
    pub fn handle(&self) -> *mut sys::NVSDK_NGX_Handle {
        self.handle.as_ptr()
    }
}

impl Drop for NgxFeature {
    fn drop(&mut self) {
        unsafe {
            if let Err(e) = sys::NVSDK_NGX_VULKAN_ReleaseFeature(self.handle.as_ptr()).result() {
                tracing::warn!(target: "ngx", "ReleaseFeature failed: {e}");
            }
        }
    }
}

/// Wraps an [`sys::NVSDK_NGX_Resource_VK`] describing a Vulkan image view that
/// NGX should read from or write to during feature evaluation.
///
/// NGX takes the resource pointer by value into its parameter map and
/// dereferences it inside `EvaluateFeature`, so `Resource` stores its descriptor
/// inline. The caller is responsible for keeping the underlying image, view,
/// and any backing memory alive at least until the command buffer recorded by
/// the `evaluate` call has completed on the GPU.
#[repr(transparent)]
pub struct Resource(sys::NVSDK_NGX_Resource_VK);

impl Resource {
    /// Builds a resource backed by a Vulkan image view.
    ///
    /// `read_write` must be `true` for output / mutated targets and `false`
    /// for read-only inputs (NGX validates this against the bound descriptor
    /// type).
    pub fn image_view(
        image_view: vk::ImageView,
        image: vk::Image,
        subresource_range: vk::ImageSubresourceRange,
        format: vk::Format,
        width: u32,
        height: u32,
        read_write: bool,
    ) -> Self {
        Self(sys::NVSDK_NGX_Resource_VK {
            Resource: sys::NVSDK_NGX_Resource_VK_Resource {
                ImageViewInfo: sys::NVSDK_NGX_ImageViewInfo_VK {
                    ImageView: image_view,
                    Image: image,
                    SubresourceRange: subresource_range,
                    Format: format,
                    Width: width,
                    Height: height,
                },
            },
            Type: sys::NVSDK_NGX_Resource_VK_Type::ImageView,
            ReadWrite: read_write,
        })
    }

    fn as_ngx_ptr(&self) -> *mut c_void {
        &self.0 as *const sys::NVSDK_NGX_Resource_VK as *mut c_void
    }
}

/// Inputs and per-frame state for one DLSS-RR evaluate call.
///
/// All `Resource` references must outlive the recorded command buffer's GPU
/// execution. `render_extent` may be smaller than the resource extent when
/// rendering at sub-output resolution; matrices, hit-distance buffers, and the
/// roughness texture are optional and should only be supplied if the engine
/// actually produces them — NGX otherwise infers from packed inputs.
pub struct DlssRrEvalDesc<'a> {
    pub color: &'a Resource,
    pub output: &'a Resource,
    pub depth: &'a Resource,
    pub motion_vectors: &'a Resource,
    pub normals: &'a Resource,
    pub diffuse_albedo: &'a Resource,
    pub specular_albedo: &'a Resource,
    pub roughness: Option<&'a Resource>,
    pub diffuse_hit_distance: Option<&'a Resource>,
    pub specular_hit_distance: Option<&'a Resource>,
    /// Sub-pixel jitter applied to this frame's projection, in pixels.
    pub jitter_offset: [f32; 2],
    /// Multiplier applied to the motion-vector texture (typically render
    /// extent if MVs are stored in NDC, or `[1.0, 1.0]` if in pixels).
    pub mv_scale: [f32; 2],
    /// Subrect of the input textures that contains valid render data.
    pub render_extent: [u32; 2],
    /// Set on the first frame and after any camera teleport / scene cut, to
    /// flush DLSS-RR's history.
    pub reset: bool,
    pub world_to_view: Option<[[f32; 4]; 4]>,
    pub view_to_clip: Option<[[f32; 4]; 4]>,
}

impl NgxFeature {
}
impl Drop for NgxContext {
    fn drop(&mut self) {
        unsafe {
            if let Err(e) = sys::NVSDK_NGX_VULKAN_Shutdown(self.device.vk_handle()).result() {
                tracing::warn!(target: "ngx", "Shutdown1 failed: {e}");
            }
        }
    }
}

unsafe extern "C" fn ngx_log_callback(
    message: *const core::ffi::c_char,
    level: sys::NVSDK_NGX_Logging_Level,
    component: sys::NVSDK_NGX_Feature,
) {
    if message.is_null() {
        return;
    }
    let msg = unsafe { CStr::from_ptr(message) }.to_string_lossy();
    let msg = msg.trim_end_matches(&['\r', '\n']);
    match level {
        sys::NVSDK_NGX_Logging_Level::Off => {}
        sys::NVSDK_NGX_Logging_Level::On => {
            tracing::info!(target: "ngx", ?component, "{msg}")
        }
        sys::NVSDK_NGX_Logging_Level::Verbose => {
            tracing::debug!(target: "ngx", ?component, "{msg}")
        }
    }
}

/// NUL-terminated `wchar_t` encoding of `path`, suitable for
/// `NVSDK_NGX_FeatureDiscoveryInfo::ApplicationDataPath`.
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
