//! Raw FFI bindings to the NVIDIA NGX SDK, limited to what DLSS Ray
//! Reconstruction on Vulkan needs. NVIDIA-named items mirror the C API, so the
//! documentation in the SDK's `nvsdk_ngx_*.h` headers applies directly. The
//! `dust_ngx_*` functions are defined in `ngx_helpers.c`.

#![allow(
    dead_code,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals
)]

use core::ffi::{c_char, c_int, c_uint};
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
            Length: 0,
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

/// `NVSDK_NGX_Feature_Support_Result::Supported`. `0` means supported; any other
/// value is a bitmask of unsupported reasons — so `FeatureSupported` is modeled
/// as a raw `c_int`, not a Rust enum (an OR of flags would be an invalid variant).
pub const NVSDK_NGX_FeatureSupportResult_Supported: c_int = 0;

/// `NVSDK_NGX_FeatureRequirement` — output of
/// [`NVSDK_NGX_VULKAN_GetFeatureRequirements`].
#[repr(C)]
#[derive(Copy, Clone)]
pub struct NVSDK_NGX_FeatureRequirement {
    /// `NVSDK_NGX_Feature_Support_Result`: `0` (Supported) or an OR of reasons.
    pub FeatureSupported: c_int,
    pub MinHWArchitecture: c_uint,
    pub MinOSVersion: [c_char; 255],
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
// DLSS-D Evaluate params
// ============================================================================

/// Slots in [`NVSDK_NGX_VK_GBuffer::pInAttrib`]; from `NVSDK_NGX_GBufferType`
/// in `nvsdk_ngx_defs.h`. The array is sized to `NVSDK_NGX_GBUFFERTYPE_NUM`
/// (16) so the C ABI matches even when callers leave most slots null.
pub const NVSDK_NGX_GBUFFERTYPE_NUM: usize = 16;

#[repr(C)]
pub struct NVSDK_NGX_VK_GBuffer {
    pub pInAttrib: [*mut NVSDK_NGX_Resource_VK; NVSDK_NGX_GBUFFERTYPE_NUM],
}

#[repr(u32)]
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum NVSDK_NGX_ToneMapperType {
    String = 0,
    Reinhard = 1,
    OneOverLuma = 2,
    Aces = 3,
}

/// `NVSDK_NGX_VK_DLSSD_Eval_Params` — see `nvsdk_ngx_helpers_dlssd_vk.h`.
///
/// Layout must match the C struct exactly: every pointer/coordinate slot is
/// preserved even when unused, because the C helper macro reads each one
/// unconditionally and forwards it into the parameter map.
///
/// Construct via `unsafe { std::mem::zeroed() }` (or
/// [`NVSDK_NGX_VK_DLSSD_Eval_Params::zeroed`]) and fill only the fields you
/// need — every field is either a nullable pointer, a numeric "leave at 0
/// for default" value, or a `Coordinates` whose 0/0 origin is the documented
/// default.
#[repr(C)]
pub struct NVSDK_NGX_VK_DLSSD_Eval_Params {
    pub pInDiffuseAlbedo: *mut NVSDK_NGX_Resource_VK,
    pub pInSpecularAlbedo: *mut NVSDK_NGX_Resource_VK,
    pub pInNormals: *mut NVSDK_NGX_Resource_VK,
    pub pInRoughness: *mut NVSDK_NGX_Resource_VK,

    pub pInColor: *mut NVSDK_NGX_Resource_VK,
    pub pInAlpha: *mut NVSDK_NGX_Resource_VK,
    pub pInOutput: *mut NVSDK_NGX_Resource_VK,
    pub pInOutputAlpha: *mut NVSDK_NGX_Resource_VK,
    pub pInDepth: *mut NVSDK_NGX_Resource_VK,
    pub pInMotionVectors: *mut NVSDK_NGX_Resource_VK,
    pub InJitterOffsetX: f32,
    pub InJitterOffsetY: f32,
    pub InRenderSubrectDimensions: NVSDK_NGX_Dimensions,

    pub InReset: c_int,
    pub InMVScaleX: f32,
    pub InMVScaleY: f32,
    pub pInTransparencyMask: *mut NVSDK_NGX_Resource_VK,
    pub pInExposureTexture: *mut NVSDK_NGX_Resource_VK,
    pub pInBiasCurrentColorMask: *mut NVSDK_NGX_Resource_VK,
    pub InAlphaSubrectBase: NVSDK_NGX_Coordinates,
    pub InOutputAlphaSubrectBase: NVSDK_NGX_Coordinates,
    pub InDiffuseAlbedoSubrectBase: NVSDK_NGX_Coordinates,
    pub InSpecularAlbedoSubrectBase: NVSDK_NGX_Coordinates,
    pub InNormalsSubrectBase: NVSDK_NGX_Coordinates,
    pub InRoughnessSubrectBase: NVSDK_NGX_Coordinates,
    pub InColorSubrectBase: NVSDK_NGX_Coordinates,
    pub InDepthSubrectBase: NVSDK_NGX_Coordinates,
    pub InMVSubrectBase: NVSDK_NGX_Coordinates,
    pub InTranslucencySubrectBase: NVSDK_NGX_Coordinates,
    pub InBiasCurrentColorSubrectBase: NVSDK_NGX_Coordinates,
    pub InOutputSubrectBase: NVSDK_NGX_Coordinates,
    pub InPreExposure: f32,
    pub InExposureScale: f32,
    pub InIndicatorInvertXAxis: c_int,
    pub InIndicatorInvertYAxis: c_int,

    pub pInReflectedAlbedo: *mut NVSDK_NGX_Resource_VK,
    pub pInColorBeforeParticles: *mut NVSDK_NGX_Resource_VK,
    pub pInColorAfterParticles: *mut NVSDK_NGX_Resource_VK,
    pub pInColorBeforeTransparency: *mut NVSDK_NGX_Resource_VK,
    pub pInColorAfterTransparency: *mut NVSDK_NGX_Resource_VK,
    pub pInColorBeforeFog: *mut NVSDK_NGX_Resource_VK,
    pub pInColorAfterFog: *mut NVSDK_NGX_Resource_VK,
    pub pInScreenSpaceSubsurfaceScatteringGuide: *mut NVSDK_NGX_Resource_VK,
    pub pInColorBeforeScreenSpaceSubsurfaceScattering: *mut NVSDK_NGX_Resource_VK,
    pub pInColorAfterScreenSpaceSubsurfaceScattering: *mut NVSDK_NGX_Resource_VK,
    pub pInScreenSpaceRefractionGuide: *mut NVSDK_NGX_Resource_VK,
    pub pInColorBeforeScreenSpaceRefraction: *mut NVSDK_NGX_Resource_VK,
    pub pInColorAfterScreenSpaceRefraction: *mut NVSDK_NGX_Resource_VK,
    pub pInDepthOfFieldGuide: *mut NVSDK_NGX_Resource_VK,
    pub pInColorBeforeDepthOfField: *mut NVSDK_NGX_Resource_VK,
    pub pInColorAfterDepthOfField: *mut NVSDK_NGX_Resource_VK,
    pub pInDiffuseHitDistance: *mut NVSDK_NGX_Resource_VK,
    pub pInSpecularHitDistance: *mut NVSDK_NGX_Resource_VK,
    pub pInDiffuseRayDirection: *mut NVSDK_NGX_Resource_VK,
    pub pInSpecularRayDirection: *mut NVSDK_NGX_Resource_VK,
    pub pInDiffuseRayDirectionHitDistance: *mut NVSDK_NGX_Resource_VK,
    pub pInSpecularRayDirectionHitDistance: *mut NVSDK_NGX_Resource_VK,
    pub InReflectedAlbedoSubrectBase: NVSDK_NGX_Coordinates,
    pub InColorBeforeParticlesSubrectBase: NVSDK_NGX_Coordinates,
    pub InColorAfterParticlesSubrectBase: NVSDK_NGX_Coordinates,
    pub InColorBeforeTransparencySubrectBase: NVSDK_NGX_Coordinates,
    pub InColorAfterTransparencySubrectBase: NVSDK_NGX_Coordinates,
    pub InColorBeforeFogSubrectBase: NVSDK_NGX_Coordinates,
    pub InColorAfterFogSubrectBase: NVSDK_NGX_Coordinates,
    pub InScreenSpaceSubsurfaceScatteringGuideSubrectBase: NVSDK_NGX_Coordinates,
    pub InColorBeforeScreenSpaceSubsurfaceScatteringSubrectBase: NVSDK_NGX_Coordinates,
    pub InColorAfterScreenSpaceSubsurfaceScatteringSubrectBase: NVSDK_NGX_Coordinates,
    pub InScreenSpaceRefractionGuideSubrectBase: NVSDK_NGX_Coordinates,
    pub InColorBeforeScreenSpaceRefractionSubrectBase: NVSDK_NGX_Coordinates,
    pub InColorAfterScreenSpaceRefractionSubrectBase: NVSDK_NGX_Coordinates,
    pub InDepthOfFieldGuideSubrectBase: NVSDK_NGX_Coordinates,
    pub InColorBeforeDepthOfFieldSubrectBase: NVSDK_NGX_Coordinates,
    pub InColorAfterDepthOfFieldSubrectBase: NVSDK_NGX_Coordinates,
    pub InDiffuseHitDistanceSubrectBase: NVSDK_NGX_Coordinates,
    pub InSpecularHitDistanceSubrectBase: NVSDK_NGX_Coordinates,
    pub InDiffuseRayDirectionSubrectBase: NVSDK_NGX_Coordinates,
    pub InSpecularRayDirectionSubrectBase: NVSDK_NGX_Coordinates,
    pub InDiffuseRayDirectionHitDistanceSubrectBase: NVSDK_NGX_Coordinates,
    pub InSpecularRayDirectionHitDistanceSubrectBase: NVSDK_NGX_Coordinates,
    pub pInWorldToViewMatrix: *mut f32,
    pub pInViewToClipMatrix: *mut f32,

    pub GBufferSurface: NVSDK_NGX_VK_GBuffer,
    pub InToneMapperType: NVSDK_NGX_ToneMapperType,
    pub pInMotionVectors3D: *mut NVSDK_NGX_Resource_VK,
    pub pInIsParticleMask: *mut NVSDK_NGX_Resource_VK,
    pub pInAnimatedTextureMask: *mut NVSDK_NGX_Resource_VK,
    pub pInDepthHighRes: *mut NVSDK_NGX_Resource_VK,
    pub pInPositionViewSpace: *mut NVSDK_NGX_Resource_VK,
    pub InFrameTimeDeltaInMsec: f32,
    pub pInRayTracingHitDistance: *mut NVSDK_NGX_Resource_VK,
    pub pInMotionVectorsReflections: *mut NVSDK_NGX_Resource_VK,
    pub pInTransparencyLayer: *mut NVSDK_NGX_Resource_VK,
    pub InTransparencyLayerSubrectBase: NVSDK_NGX_Coordinates,
    pub pInTransparencyLayerOpacity: *mut NVSDK_NGX_Resource_VK,
    pub InTransparencyLayerOpacitySubrectBase: NVSDK_NGX_Coordinates,
    pub pInTransparencyLayerMvecs: *mut NVSDK_NGX_Resource_VK,
    pub InTransparencyLayerMvecsSubrectBase: NVSDK_NGX_Coordinates,
    pub pInDisocclusionMask: *mut NVSDK_NGX_Resource_VK,
    pub InDisocclusionMaskSubrectBase: NVSDK_NGX_Coordinates,
}

impl NVSDK_NGX_VK_DLSSD_Eval_Params {
    /// Zero-initialised eval params — every pointer NULL, every coordinate
    /// (0, 0), every numeric 0. Matches the documented "leave to 0/0.0f if
    /// unused" defaults of the C struct.
    pub fn zeroed() -> Self {
        unsafe { core::mem::zeroed() }
    }
}

// ============================================================================
// Parameter map (opaque)
// ============================================================================

/// Opaque NGX parameter map.
#[repr(C)]
pub struct NVSDK_NGX_Parameter {
    _private: [u8; 0],
}

// ============================================================================
// extern "C" entry points
// ============================================================================

unsafe extern "C" {
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

    pub fn NVSDK_NGX_VULKAN_DestroyParameters(
        InParameters: *mut NVSDK_NGX_Parameter,
    ) -> NVSDK_NGX_Result;

    // ---- Feature lifecycle ----
    pub fn NVSDK_NGX_VULKAN_ReleaseFeature(InHandle: *mut NVSDK_NGX_Handle) -> NVSDK_NGX_Result;

    /// C wrapper around the `static inline` `NGX_VULKAN_CREATE_DLSSD_EXT1`
    /// helper from `nvsdk_ngx_helpers_dlssd_vk.h`.
    pub fn dust_ngx_vulkan_create_dlssd_ext1(
        InDevice: vk::Device,
        InCmdList: vk::CommandBuffer,
        InCreationNodeMask: c_uint,
        InVisibilityNodeMask: c_uint,
        ppOutHandle: *mut *mut NVSDK_NGX_Handle,
        pInParams: *mut NVSDK_NGX_Parameter,
        pInDlssDCreateParams: *const NVSDK_NGX_DLSSD_Create_Params,
    ) -> NVSDK_NGX_Result;

    /// C wrapper around the `static inline` `NGX_VULKAN_EVALUATE_DLSSD_EXT`
    /// helper from `nvsdk_ngx_helpers_dlssd_vk.h`.
    pub fn dust_ngx_vulkan_evaluate_dlssd_ext(
        InCmdList: vk::CommandBuffer,
        pInHandle: *mut NVSDK_NGX_Handle,
        pInParams: *mut NVSDK_NGX_Parameter,
        pInDlssDEvalParams: *mut NVSDK_NGX_VK_DLSSD_Eval_Params,
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

    /// Pre-init discovery: reports whether a feature is supported on the given
    /// adapter *without* calling `NVSDK_NGX_VULKAN_Init`. Needs only a
    /// `VkInstance` + `VkPhysicalDevice`; `OutSupported` is written on success.
    pub fn NVSDK_NGX_VULKAN_GetFeatureRequirements(
        Instance: vk::Instance,
        PhysicalDevice: vk::PhysicalDevice,
        FeatureDiscoveryInfo: *const NVSDK_NGX_FeatureDiscoveryInfo,
        OutSupported: *mut NVSDK_NGX_FeatureRequirement,
    ) -> NVSDK_NGX_Result;
}
