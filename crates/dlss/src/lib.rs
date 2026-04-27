//! Safe wrapper around the NVIDIA DLSS NGX SDK (Vulkan + Ray Reconstruction).
//!
//! Stage 1 surface: just `DlssError` + a result alias. Real types
//! (`NgxContext`, `DlssRrFeature`, `Resource`) land in later stages once the
//! engine has motion vectors, jitter, and split render/output resolutions.

use pumicite::utils::AsVkHandle;
use pumicite::Device;
use std::ffi::{c_int, c_uint, c_void, c_ulonglong};
use std::mem::MaybeUninit;
use std::ptr::NonNull;
use std::{ffi::CStr, ptr};

use crate::sys::wchar_t;
pub use bevy::*;

mod bevy;
mod sys;

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

/// Owns the NGX runtime + capability parameter map for the lifetime of the app.
///
/// Inserted as a Bevy resource by [`DLSSPlugin::finish`]. Dropping the resource
/// destroys the parameter map and calls `NVSDK_NGX_VULKAN_Shutdown1`, so the
/// context must be removed before the [`Device`] it references is destroyed.
pub struct NgxContext {
    device: Device,
}

unsafe impl Send for NgxContext {}
unsafe impl Sync for NgxContext {}

impl NgxContext {
    /// Allocates the capability parameter map and probes DLSS-RR availability.
    /// Must be called only after `NVSDK_NGX_VULKAN_Init_with_ProjectID` succeeded.
    fn new(device: Device, application_data_path: &[wchar_t]) -> DlssResult<Self> {
        assert_eq!(
            application_data_path.last(),
            Some(&0),
            "application_data_path must be null-terminated"
        );
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
                    PathListInfo: sys::NVSDK_NGX_PathListInfo {
                        Path: std::ptr::null(),
                        Length: 0,
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
    pub fn allocate_parameters() -> DlssResult<ParameterMap> {
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
    pub fn get_capability_parameters() -> DlssResult<ParameterMap> {
        let mut parameters: *mut sys::NVSDK_NGX_Parameter = ptr::null_mut();
        unsafe {
            sys::NVSDK_NGX_VULKAN_GetCapabilityParameters(&mut parameters).result()?;
        }
        Ok(ParameterMap(NonNull::new(parameters).unwrap()))
    }
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
