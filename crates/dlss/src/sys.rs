//! Raw FFI bindings to the NVIDIA DLSS NGX SDK (Vulkan + DLSS-D / RayReconstruction).
//!
//! Only the subset used by `dust_dlss` is bound here. Symbol names mirror the C
//! API exactly so the upstream documentation in `nvsdk_ngx_*.h` applies directly.
//!
//! Vulkan handle and enum types are re-exported from `ash` so callers don't need
//! to convert between FFI primitive types and ash wrappers.

#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]

use core::ffi::{c_char, c_int, c_void};
use std::ffi::{c_uint, c_ulonglong};
use pumicite::ash::vk;

/// `wchar_t` — 16-bit on Windows (UTF-16), 32-bit on Linux (UTF-32).
/// NGX takes paths as `wchar_t*`, so callers must pre-encode strings.
#[cfg(target_os = "windows")]
pub type wchar_t = u16;
#[cfg(not(target_os = "windows"))]
pub type wchar_t = i32;

// ============================================================================
// Constants
// ============================================================================

/// `NVSDK_NGX_VERSION_API_MACRO` — passed as `InSDKVersion` to Init.
pub const NVSDK_NGX_VERSION_API: u32 = 0x0000015;

// ============================================================================
// Result codes
// ============================================================================

/// `NVSDK_NGX_Result` — see `nvsdk_ngx_defs.h`.
#[repr(transparent)]
#[derive(Copy, Clone, Eq, PartialEq)]
pub struct NVSDK_NGX_Result(pub u32);

impl NVSDK_NGX_Result {
    pub const Success: Self = Self(0x1);
    pub const Fail: Self = Self(0xBAD0_0000);
    pub const FAIL_FeatureNotSupported: Self = Self(0xBAD0_0000 | 1);
    pub const FAIL_PlatformError: Self = Self(0xBAD0_0000 | 2);
    pub const FAIL_FeatureAlreadyExists: Self = Self(0xBAD0_0000 | 3);
    pub const FAIL_FeatureNotFound: Self = Self(0xBAD0_0000 | 4);
    pub const FAIL_InvalidParameter: Self = Self(0xBAD0_0000 | 5);
    pub const FAIL_ScratchBufferTooSmall: Self = Self(0xBAD0_0000 | 6);
    pub const FAIL_NotInitialized: Self = Self(0xBAD0_0000 | 7);
    pub const FAIL_UnsupportedInputFormat: Self = Self(0xBAD0_0000 | 8);
    pub const FAIL_RWFlagMissing: Self = Self(0xBAD0_0000 | 9);
    pub const FAIL_MissingInput: Self = Self(0xBAD0_0000 | 10);
    pub const FAIL_UnableToInitializeFeature: Self = Self(0xBAD0_0000 | 11);
    pub const FAIL_OutOfDate: Self = Self(0xBAD0_0000 | 12);
    pub const FAIL_OutOfGPUMemory: Self = Self(0xBAD0_0000 | 13);
    pub const FAIL_UnsupportedFormat: Self = Self(0xBAD0_0000 | 14);
    pub const FAIL_UnableToWriteToAppDataPath: Self = Self(0xBAD0_0000 | 15);
    pub const FAIL_UnsupportedParameter: Self = Self(0xBAD0_0000 | 16);
    pub const FAIL_Denied: Self = Self(0xBAD0_0000 | 17);
    pub const FAIL_NotImplemented: Self = Self(0xBAD0_0000 | 18);

    #[inline]
    pub fn succeeded(self) -> bool {
        (self.0 & 0xFFF0_0000) != Self::Fail.0
    }

    #[inline]
    pub fn failed(self) -> bool {
        !self.succeeded()
    }
}

// ============================================================================
// Enums
// ============================================================================

#[repr(u32)]
#[non_exhaustive]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum NVSDK_NGX_Feature {
    Reserved0 = 0,
    SuperSampling = 1,
    InPainting = 2,
    ImageSuperResolution = 3,
    SlowMotion = 4,
    VideoSuperResolution = 5,
    Reserved1 = 6,
    Reserved2 = 7,
    Reserved3 = 8,
    ImageSignalProcessing = 9,
    DeepResolve = 10,
    FrameGeneration = 11,
    DeepDVC = 12,
    RayReconstruction = 13,
    Reserved14 = 14,
    Reserved15 = 15,
    Reserved16 = 16,
    // Count = 17,
    Reserved_SDK = 32764,
    Reserved_Core = 32765,
    Reserved_Unknown = 32766,
}

#[repr(i32)]
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum NVSDK_NGX_PerfQuality_Value {
    MaxPerf = 0,
    Balanced = 1,
    MaxQuality = 2,
    UltraPerformance = 3,
    UltraQuality = 4,
    DLAA = 5,
}

#[repr(i32)]
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum NVSDK_NGX_DLSS_Denoise_Mode {
    Off = 0,
    DLUnified = 1,
}

#[repr(i32)]
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum NVSDK_NGX_DLSS_Roughness_Mode {
    Unpacked = 0,
    /// Roughness is read from `normals.w`.
    Packed = 1,
}

#[repr(i32)]
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum NVSDK_NGX_DLSS_Depth_Type {
    Linear = 0,
    HW = 1,
}

#[repr(i32)]
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum NVSDK_NGX_EngineType {
    Custom = 0,
    Unreal = 1,
    Unity = 2,
    Omniverse = 3,
}

#[repr(i32)]
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum NVSDK_NGX_Logging_Level {
    Off = 0,
    On = 1,
    Verbose = 2,
}

#[repr(i32)]
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum NVSDK_NGX_Application_Identifier_Type {
    ApplicationId = 0,
    ProjectId = 1,
}

#[repr(i32)]
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum NVSDK_NGX_Resource_VK_Type {
    ImageView = 0,
    Buffer = 1,
}

/// Bitfield — values OR'd into `NVSDK_NGX_DLSSD_Create_Params::InFeatureCreateFlags`.
pub mod NVSDK_NGX_DLSS_Feature_Flags {
    pub const None: i32 = 0;
    pub const IsHDR: i32 = 1 << 0;
    pub const MVLowRes: i32 = 1 << 1;
    pub const MVJittered: i32 = 1 << 2;
    pub const DepthInverted: i32 = 1 << 3;
    pub const AutoExposure: i32 = 1 << 6;
    pub const AlphaUpscaling: i32 = 1 << 7;
}

// ============================================================================
// Small structs
// ============================================================================

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct NVSDK_NGX_Handle {
    pub Id: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct NVSDK_NGX_Coordinates {
    pub X: u32,
    pub Y: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct NVSDK_NGX_Dimensions {
    pub Width: u32,
    pub Height: u32,
}

// ============================================================================
// Resource_VK
// ============================================================================

#[repr(C)]
#[derive(Copy, Clone)]
pub struct NVSDK_NGX_ImageViewInfo_VK {
    pub ImageView: vk::ImageView,
    pub Image: vk::Image,
    pub SubresourceRange: vk::ImageSubresourceRange,
    pub Format: vk::Format,
    pub Width: u32,
    pub Height: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct NVSDK_NGX_BufferInfo_VK {
    pub Buffer: vk::Buffer,
    pub SizeInBytes: u32,
}

#[repr(C)]
pub union NVSDK_NGX_Resource_VK_Resource {
    pub ImageViewInfo: NVSDK_NGX_ImageViewInfo_VK,
    pub BufferInfo: NVSDK_NGX_BufferInfo_VK,
}

#[repr(C)]
pub struct NVSDK_NGX_Resource_VK {
    pub Resource: NVSDK_NGX_Resource_VK_Resource,
    pub Type: NVSDK_NGX_Resource_VK_Type,
    pub ReadWrite: bool,
}

// ============================================================================
// Application identifier + feature discovery
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NVSDK_NGX_ProjectIdDescription {
    pub ProjectId: *const c_char,
    pub EngineType: NVSDK_NGX_EngineType,
    pub EngineVersion: *const c_char,
}

#[repr(C)]
pub union NVSDK_NGX_Application_Identifier_Value {
    pub ProjectDesc: NVSDK_NGX_ProjectIdDescription,
    pub ApplicationId: u64,
}

#[repr(C)]
pub struct NVSDK_NGX_Application_Identifier {
    pub IdentifierType: NVSDK_NGX_Application_Identifier_Type,
    pub v: NVSDK_NGX_Application_Identifier_Value,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct NVSDK_NGX_PathListInfo {
    pub Path: *const *const wchar_t,
    pub Length: u32,
}
impl Default for NVSDK_NGX_PathListInfo {
    fn default() -> Self {
        Self {
            Path: std::ptr::null(),
            Length: 0
        }
    }
}

/// Opaque internal NGX state.
#[repr(C)]
pub struct NVSDK_NGX_FeatureCommonInfo_Internal {
    _private: [u8; 0],
}

pub type NVSDK_NGX_AppLogCallback = unsafe extern "C" fn(
    message: *const c_char,
    loggingLevel: NVSDK_NGX_Logging_Level,
    sourceComponent: NVSDK_NGX_Feature,
);

#[repr(C)]
pub struct NVSDK_NGX_LoggingInfo {
    pub LoggingCallback: Option<NVSDK_NGX_AppLogCallback>,
    pub MinimumLoggingLevel: NVSDK_NGX_Logging_Level,
    pub DisableOtherLoggingSinks: bool,
}

#[repr(C)]
pub struct NVSDK_NGX_FeatureCommonInfo {
    pub PathListInfo: NVSDK_NGX_PathListInfo,
    pub InternalData: *mut NVSDK_NGX_FeatureCommonInfo_Internal,
    pub LoggingInfo: NVSDK_NGX_LoggingInfo,
}

#[repr(C)]
pub struct NVSDK_NGX_FeatureDiscoveryInfo {
    /// Always `NVSDK_NGX_VERSION_API`.
    pub SDKVersion: u32,
    pub FeatureID: NVSDK_NGX_Feature,
    pub Identifier: NVSDK_NGX_Application_Identifier,
    /// `wchar_t*` — UTF-16 on Windows, UTF-32 on Linux.
    pub ApplicationDataPath: *const wchar_t,
    pub FeatureInfo: *const NVSDK_NGX_FeatureCommonInfo,
}

// ============================================================================
// DLSS-D Create params
// ============================================================================

#[repr(C)]
pub struct NVSDK_NGX_DLSSD_Create_Params {
    pub InDenoiseMode: NVSDK_NGX_DLSS_Denoise_Mode,
    pub InRoughnessMode: NVSDK_NGX_DLSS_Roughness_Mode,
    pub InUseHWDepth: NVSDK_NGX_DLSS_Depth_Type,

    pub InWidth: u32,
    pub InHeight: u32,
    pub InTargetWidth: u32,
    pub InTargetHeight: u32,

    pub InPerfQualityValue: NVSDK_NGX_PerfQuality_Value,
    pub InFeatureCreateFlags: c_int,
    pub InEnableOutputSubrects: bool,
}

// ============================================================================
// Parameter map (opaque) + setters/getters
// ============================================================================

/// Opaque NGX parameter map (C++ has virtual methods; from C we use the setter
/// helper functions below).
#[repr(C)]
pub struct NVSDK_NGX_Parameter {
    _private: [u8; 0],
}

// ============================================================================
// extern "C" entry points
// ============================================================================

unsafe extern "C" {
    // ---- Parameter setters / getters ----
    pub fn NVSDK_NGX_Parameter_SetULL(
        InParameter: *mut NVSDK_NGX_Parameter,
        InName: *const c_char,
        InValue: u64,
    );
    pub fn NVSDK_NGX_Parameter_SetF(
        InParameter: *mut NVSDK_NGX_Parameter,
        InName: *const c_char,
        InValue: f32,
    );
    pub fn NVSDK_NGX_Parameter_SetD(
        InParameter: *mut NVSDK_NGX_Parameter,
        InName: *const c_char,
        InValue: f64,
    );
    pub fn NVSDK_NGX_Parameter_SetUI(
        InParameter: *mut NVSDK_NGX_Parameter,
        InName: *const c_char,
        InValue: u32,
    );
    pub fn NVSDK_NGX_Parameter_SetI(
        InParameter: *mut NVSDK_NGX_Parameter,
        InName: *const c_char,
        InValue: c_int,
    );
    pub fn NVSDK_NGX_Parameter_SetVoidPointer(
        InParameter: *mut NVSDK_NGX_Parameter,
        InName: *const c_char,
        InValue: *mut c_void,
    );

    pub fn NVSDK_NGX_Parameter_GetULL(
        InParameter: *mut NVSDK_NGX_Parameter,
        InName: *const c_char,
        OutValue: *mut c_ulonglong,
    ) -> NVSDK_NGX_Result;
    pub fn NVSDK_NGX_Parameter_GetF(
        InParameter: *mut NVSDK_NGX_Parameter,
        InName: *const c_char,
        OutValue: *mut f32,
    ) -> NVSDK_NGX_Result;
    pub fn NVSDK_NGX_Parameter_GetUI(
        InParameter: *mut NVSDK_NGX_Parameter,
        InName: *const c_char,
        OutValue: *mut c_uint,
    ) -> NVSDK_NGX_Result;
    pub fn NVSDK_NGX_Parameter_GetI(
        InParameter: *mut NVSDK_NGX_Parameter,
        InName: *const c_char,
        OutValue: *mut c_int,
    ) -> NVSDK_NGX_Result;
    pub fn NVSDK_NGX_Parameter_GetVoidPointer(
        InParameter: *mut NVSDK_NGX_Parameter,
        InName: *const c_char,
        OutValue: *mut *mut c_void,
    ) -> NVSDK_NGX_Result;

    // ---- Vulkan init / shutdown ----
    pub fn NVSDK_NGX_VULKAN_Init_with_ProjectID(
        InProjectId: *const c_char,
        InEngineType: NVSDK_NGX_EngineType,
        InEngineVersion: *const c_char,
        InApplicationDataPath: *const wchar_t,
        InInstance: vk::Instance,
        InPD: vk::PhysicalDevice,
        InDevice: vk::Device,
        InGIPA: Option<vk::PFN_vkGetInstanceProcAddr>,
        InGDPA: Option<vk::PFN_vkGetDeviceProcAddr>,
        InFeatureInfo: *const NVSDK_NGX_FeatureCommonInfo,
        InSDKVersion: u32,
    ) -> NVSDK_NGX_Result;

    pub fn NVSDK_NGX_VULKAN_Shutdown(InDevice: vk::Device) -> NVSDK_NGX_Result;

    // ---- Parameter-map lifecycle ----
    pub fn NVSDK_NGX_VULKAN_AllocateParameters(
        OutParameters: *mut *mut NVSDK_NGX_Parameter,
    ) -> NVSDK_NGX_Result;

    pub fn NVSDK_NGX_VULKAN_GetCapabilityParameters(
        OutParameters: *mut *mut NVSDK_NGX_Parameter,
    ) -> NVSDK_NGX_Result;

    pub fn NVSDK_NGX_VULKAN_DestroyParameters(
        InParameters: *mut NVSDK_NGX_Parameter,
    ) -> NVSDK_NGX_Result;

    // ---- Feature lifecycle ----
    pub fn NVSDK_NGX_VULKAN_CreateFeature1(
        InDevice: vk::Device,
        InCmdList: vk::CommandBuffer,
        InFeatureID: NVSDK_NGX_Feature,
        InParameters: *mut NVSDK_NGX_Parameter,
        OutHandle: *mut *mut NVSDK_NGX_Handle,
    ) -> NVSDK_NGX_Result;

    pub fn NVSDK_NGX_VULKAN_ReleaseFeature(InHandle: *mut NVSDK_NGX_Handle) -> NVSDK_NGX_Result;

    /// C wrapper around the `static inline` `NGX_VULKAN_CREATE_DLSSD_EXT1`
    /// helper from `nvsdk_ngx_helpers_dlssd_vk.h`. Defined in
    /// `third_party/dlss_wrapper.c` (target `//third_party:dlss_helpers`).
    pub fn dust_ngx_vulkan_create_dlssd_ext1(
        InDevice: vk::Device,
        InCmdList: vk::CommandBuffer,
        InCreationNodeMask: c_uint,
        InVisibilityNodeMask: c_uint,
        ppOutHandle: *mut *mut NVSDK_NGX_Handle,
        pInParams: *mut NVSDK_NGX_Parameter,
        pInDlssDCreateParams: *const NVSDK_NGX_DLSSD_Create_Params,
    ) -> NVSDK_NGX_Result;

    pub fn NVSDK_NGX_VULKAN_EvaluateFeature_C(
        InCmdList: vk::CommandBuffer,
        InFeatureHandle: *const NVSDK_NGX_Handle,
        InParameters: *const NVSDK_NGX_Parameter,
        InCallback: Option<unsafe extern "C" fn(progress: f32, should_cancel: *mut bool)>,
    ) -> NVSDK_NGX_Result;

    // ---- Extension requirement queries (called pre-instance/device creation) ----
    pub fn NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(
        FeatureDiscoveryInfo: *const NVSDK_NGX_FeatureDiscoveryInfo,
        OutExtensionCount: *mut u32,
        OutExtensionProperties: *mut *mut vk::ExtensionProperties,
    ) -> NVSDK_NGX_Result;

    pub fn NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(
        Instance: vk::Instance,
        PhysicalDevice: vk::PhysicalDevice,
        FeatureDiscoveryInfo: *const NVSDK_NGX_FeatureDiscoveryInfo,
        OutExtensionCount: *mut u32,
        OutExtensionProperties: *mut *mut vk::ExtensionProperties,
    ) -> NVSDK_NGX_Result;
}

// ============================================================================
// Parameter name constants
// ============================================================================
//
// NGX parameter map uses string keys. We expose only the ones used by DLSS-RR.
// All keys are NUL-terminated (the C macros expand to string literals).

pub mod params {
    macro_rules! ngx_param {
        ($name:ident, $value:literal) => {
            pub const $name: &core::ffi::CStr =
                match core::ffi::CStr::from_bytes_with_nul(concat!($value, "\0").as_bytes()) {
                    Ok(s) => s,
                    Err(_) => panic!("ngx param string contained interior NUL"),
                };
        };
    }

    // Feature creation
    ngx_param!(
        SuperSamplingDenoising_Available,
        "SuperSamplingDenoising.Available"
    );
    ngx_param!(
        SuperSamplingDenoising_NeedsUpdatedDriver,
        "SuperSamplingDenoising.NeedsUpdatedDriver"
    );
    ngx_param!(
        SuperSamplingDenoising_FeatureInitResult,
        "SuperSamplingDenoising.FeatureInitResult"
    );

    ngx_param!(CreationNodeMask, "CreationNodeMask");
    ngx_param!(VisibilityNodeMask, "VisibilityNodeMask");
    ngx_param!(Width, "Width");
    ngx_param!(Height, "Height");
    ngx_param!(OutWidth, "OutWidth");
    ngx_param!(OutHeight, "OutHeight");
    ngx_param!(PerfQualityValue, "PerfQualityValue");
    ngx_param!(DLSS_Feature_Create_Flags, "DLSS.Feature.Create.Flags");
    ngx_param!(DLSS_Enable_Output_Subrects, "DLSS.Enable.Output.Subrects");
    ngx_param!(DLSS_Denoise_Mode, "DLSS.Denoise.Mode");
    ngx_param!(DLSS_Roughness_Mode, "DLSS.Roughness.Mode");
    ngx_param!(Use_HW_Depth, "DLSS.Use.HW.Depth");

    // Eval inputs
    ngx_param!(Color, "Color");
    ngx_param!(Output, "Output");
    ngx_param!(Depth, "Depth");
    ngx_param!(MotionVectors, "MotionVectors");
    ngx_param!(GBuffer_Normals, "GBuffer.Normals");
    ngx_param!(DiffuseAlbedo, "DLSS.Input.DiffuseAlbedo");
    ngx_param!(SpecularAlbedo, "DLSS.Input.SpecularAlbedo");
    ngx_param!(GBuffer_Roughness, "GBuffer.Roughness");
    ngx_param!(Reset, "Reset");
    ngx_param!(MV_Scale_X, "MV.Scale.X");
    ngx_param!(MV_Scale_Y, "MV.Scale.Y");
    ngx_param!(Jitter_Offset_X, "Jitter.Offset.X");
    ngx_param!(Jitter_Offset_Y, "Jitter.Offset.Y");
    ngx_param!(
        DLSS_Render_Subrect_Dimensions_Width,
        "DLSS.Render.Subrect.Dimensions.Width"
    );
    ngx_param!(
        DLSS_Render_Subrect_Dimensions_Height,
        "DLSS.Render.Subrect.Dimensions.Height"
    );

    // Hit-distance + matrices (optional — used for reflection quality)
    ngx_param!(DLSSD_DiffuseHitDistance, "DLSSD.DiffuseHitDistance");
    ngx_param!(DLSSD_SpecularHitDistance, "DLSSD.SpecularHitDistance");
    ngx_param!(DLSS_WORLD_TO_VIEW_MATRIX, "WorldToViewMatrix");
    ngx_param!(DLSS_VIEW_TO_CLIP_MATRIX, "ViewToClipMatrix");
}
