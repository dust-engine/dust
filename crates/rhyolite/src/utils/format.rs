use ash::vk;
use glam::Vec2;
use serde::{Deserialize, Serialize};

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum FormatType {
    /// Value will be converted to a float in the range of [0, 1]
    UNorm,
    /// Value will be converted to as a float in the range of [-1, 1]
    SNorm,
    /// Value will be intepreted as an unsigned integer, then cast to a float with the same magnitude.
    /// For example, R8_USCALED will be converted to a float in the range of [0, 255]
    UScaled,
    /// Value will be intepreted as a signed integer, then cast to a float with the same magnitude.
    /// For example, R8_SSCALED will be converted to a float in the range of [-128, 127]
    SScaled,
    /// Value will be directly interpreted as an integer in the range of [0, 255]
    UInt,
    /// Value will be directly interpreted as an integer in the range of [-128, 127]
    SInt,

    sRGB,
    SFloat,
    UFloat,
}

#[derive(Debug, Clone)]
pub struct Format {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
    pub ty: FormatType,
    pub permutation: Permutation,
}

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, Serialize, Deserialize, Copy)]
pub enum Permutation {
    R,
    G,
    B,
    RG,
    RGB,
    BGR,
    RGBA,
    BGRA,
    ARGB,
    ABGR,

    /// A three-component format with shared exponent.
    EBGR,

    /// Depth
    D,
    /// Stencil
    S,
    /// Depth Stencil
    DS,

    /// Each 64-bit compressed texel block encodes a 4x4 rectangle of unsigned normalized RGB texel data.
    BC1_RGB,
    /// Each 64-bit compressed texel block encodes a 4x4 rectangle of unsigned normalized RGB texel data, and provides 1 bit of alpha.
    BC1_RGBA,

    BC2,
    BC3,
    BC4,
    BC5,
    BC6H,
    BC7,
    ETC2_RGB,
    ETC2_RGBA,
    EAC_R,
    EAC_RG,
    ASTC {
        x: u8,
        y: u8,
    },
}

impl From<vk::Format> for Format {
    #[rustfmt::skip]
    fn from(value: vk::Format) -> Self {
        match value {
            vk::Format::R4G4_UNORM_PACK8 => Format { r: 4, g: 4, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::RG },
            vk::Format::R4G4B4A4_UNORM_PACK16 => Format { r: 4, g: 4, b: 4, a: 4, ty: FormatType::UNorm, permutation: Permutation::RGBA },
            vk::Format::B4G4R4A4_UNORM_PACK16 => Format { r: 4, g: 4, b: 4, a: 4, ty: FormatType::UNorm, permutation: Permutation::BGRA },
            vk::Format::R5G6B5_UNORM_PACK16 => Format { r: 5, g: 6, b: 5, a: 0, ty: FormatType::UNorm, permutation: Permutation::RGB },
            vk::Format::B5G6R5_UNORM_PACK16 => Format { r: 5, g: 6, b: 5, a: 0, ty: FormatType::UNorm, permutation: Permutation::BGR },
            vk::Format::R5G5B5A1_UNORM_PACK16 => Format { r: 5, g: 5, b: 5, a: 1, ty: FormatType::UNorm, permutation: Permutation::RGBA },
            vk::Format::B5G5R5A1_UNORM_PACK16 => Format { r: 5, g: 5, b: 5, a: 1, ty: FormatType::UNorm, permutation: Permutation::BGRA },
            vk::Format::A1R5G5B5_UNORM_PACK16 => Format { r: 5, g: 5, b: 5, a: 1, ty: FormatType::UNorm, permutation: Permutation::ARGB },

            vk::Format::R8_UNORM => Format { r: 8, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::R },
            vk::Format::R8_SNORM => Format { r: 8, g: 0, b: 0, a: 0, ty: FormatType::SNorm, permutation: Permutation::R },
            vk::Format::R8_USCALED => Format { r: 8, g: 0, b: 0, a: 0, ty: FormatType::UScaled, permutation: Permutation::R },
            vk::Format::R8_SSCALED => Format { r: 8, g: 0, b: 0, a: 0, ty: FormatType::SScaled, permutation: Permutation::R },
            vk::Format::R8_UINT => Format { r: 8, g: 0, b: 0, a: 0, ty: FormatType::UInt, permutation: Permutation::R },
            vk::Format::R8_SINT => Format { r: 8, g: 0, b: 0, a: 0, ty: FormatType::SInt, permutation: Permutation::R },
            vk::Format::R8_SRGB => Format { r: 8, g: 0, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::R },

            vk::Format::R8G8_UNORM => Format { r: 8, g: 8, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::RG },
            vk::Format::R8G8_SNORM => Format { r: 8, g: 8, b: 0, a: 0, ty: FormatType::SNorm, permutation: Permutation::RG },
            vk::Format::R8G8_USCALED => Format { r: 8, g: 8, b: 0, a: 0, ty: FormatType::UScaled, permutation: Permutation::RG },
            vk::Format::R8G8_SSCALED => Format { r: 8, g: 8, b: 0, a: 0, ty: FormatType::SScaled, permutation: Permutation::RG },
            vk::Format::R8G8_UINT => Format { r: 8, g: 8, b: 0, a: 0, ty: FormatType::UInt, permutation: Permutation::RG },
            vk::Format::R8G8_SINT => Format { r: 8, g: 8, b: 0, a: 0, ty: FormatType::SInt, permutation: Permutation::RG },
            vk::Format::R8G8_SRGB => Format { r: 8, g: 8, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::RG },

            vk::Format::R8G8B8_UNORM => Format { r: 8, g: 8, b: 8, a: 0, ty: FormatType::UNorm, permutation: Permutation::RGB },
            vk::Format::R8G8B8_SNORM => Format { r: 8, g: 8, b: 8, a: 0, ty: FormatType::SNorm, permutation: Permutation::RGB },
            vk::Format::R8G8B8_USCALED => Format { r: 8, g: 8, b: 8, a: 0, ty: FormatType::UScaled, permutation: Permutation::RGB },
            vk::Format::R8G8B8_SSCALED => Format { r: 8, g: 8, b: 8, a: 0, ty: FormatType::SScaled, permutation: Permutation::RGB },
            vk::Format::R8G8B8_UINT => Format { r: 8, g: 8, b: 8, a: 0, ty: FormatType::UInt, permutation: Permutation::RGB },
            vk::Format::R8G8B8_SINT => Format { r: 8, g: 8, b: 8, a: 0, ty: FormatType::SInt, permutation: Permutation::RGB },
            vk::Format::R8G8B8_SRGB => Format { r: 8, g: 8, b: 8, a: 0, ty: FormatType::sRGB, permutation: Permutation::RGB },

            vk::Format::B8G8R8_UNORM => Format { r: 8, g: 8, b: 8, a: 0, ty: FormatType::UNorm, permutation: Permutation::BGR },
            vk::Format::B8G8R8_SNORM => Format { r: 8, g: 8, b: 8, a: 0, ty: FormatType::SNorm, permutation: Permutation::BGR },
            vk::Format::B8G8R8_USCALED => Format { r: 8, g: 8, b: 8, a: 0, ty: FormatType::UScaled, permutation: Permutation::BGR },
            vk::Format::B8G8R8_SSCALED => Format { r: 8, g: 8, b: 8, a: 0, ty: FormatType::SScaled, permutation: Permutation::BGR },
            vk::Format::B8G8R8_UINT => Format { r: 8, g: 8, b: 8, a: 0, ty: FormatType::UInt, permutation: Permutation::BGR },
            vk::Format::B8G8R8_SINT => Format { r: 8, g: 8, b: 8, a: 0, ty: FormatType::SInt, permutation: Permutation::BGR },
            vk::Format::B8G8R8_SRGB => Format { r: 8, g: 8, b: 8, a: 0, ty: FormatType::sRGB, permutation: Permutation::BGR },

            vk::Format::R8G8B8A8_UNORM => Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::UNorm, permutation: Permutation::RGBA },
            vk::Format::R8G8B8A8_SNORM => Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::SNorm, permutation: Permutation::RGBA },
            vk::Format::R8G8B8A8_USCALED => Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::UScaled, permutation: Permutation::RGBA },
            vk::Format::R8G8B8A8_SSCALED => Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::SScaled, permutation: Permutation::RGBA },
            vk::Format::R8G8B8A8_UINT => Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::UInt, permutation: Permutation::RGBA },
            vk::Format::R8G8B8A8_SINT => Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::SInt, permutation: Permutation::RGBA },
            vk::Format::R8G8B8A8_SRGB => Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::sRGB, permutation: Permutation::RGBA },

            vk::Format::B8G8R8A8_UNORM => Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::UNorm, permutation: Permutation::BGRA },
            vk::Format::B8G8R8A8_SNORM => Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::SNorm, permutation: Permutation::BGRA },
            vk::Format::B8G8R8A8_USCALED => Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::UScaled, permutation: Permutation::BGRA },
            vk::Format::B8G8R8A8_SSCALED => Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::SScaled, permutation: Permutation::BGRA },
            vk::Format::B8G8R8A8_UINT => Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::UInt, permutation: Permutation::BGRA },
            vk::Format::B8G8R8A8_SINT => Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::SInt, permutation: Permutation::BGRA },
            vk::Format::B8G8R8A8_SRGB => Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::sRGB, permutation: Permutation::BGRA },

            vk::Format::A8B8G8R8_UNORM_PACK32 => Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::UNorm, permutation: Permutation::ABGR },
            vk::Format::A8B8G8R8_SNORM_PACK32 => Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::SNorm, permutation: Permutation::ABGR },
            vk::Format::A8B8G8R8_USCALED_PACK32 => Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::UScaled, permutation: Permutation::ABGR },
            vk::Format::A8B8G8R8_SSCALED_PACK32 => Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::SScaled, permutation: Permutation::ABGR },
            vk::Format::A8B8G8R8_UINT_PACK32 => Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::UInt, permutation: Permutation::ABGR },
            vk::Format::A8B8G8R8_SINT_PACK32 => Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::SInt, permutation: Permutation::ABGR },
            vk::Format::A8B8G8R8_SRGB_PACK32 => Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::sRGB, permutation: Permutation::ABGR },

            vk::Format::A2R10G10B10_UNORM_PACK32 => Format { r: 10, g: 10, b: 10, a: 2, ty: FormatType::UNorm, permutation: Permutation::ARGB },
            vk::Format::A2R10G10B10_SNORM_PACK32 => Format { r: 10, g: 10, b: 10, a: 2, ty: FormatType::SNorm, permutation: Permutation::ARGB },
            vk::Format::A2R10G10B10_USCALED_PACK32 => Format { r: 10, g: 10, b: 10, a: 2, ty: FormatType::UScaled, permutation: Permutation::ARGB },
            vk::Format::A2R10G10B10_SSCALED_PACK32 => Format { r: 10, g: 10, b: 10, a: 2, ty: FormatType::SScaled, permutation: Permutation::ARGB },
            vk::Format::A2R10G10B10_UINT_PACK32 => Format { r: 10, g: 10, b: 10, a: 2, ty: FormatType::UInt, permutation: Permutation::ARGB },
            vk::Format::A2R10G10B10_SINT_PACK32 => Format { r: 10, g: 10, b: 10, a: 2, ty: FormatType::SInt, permutation: Permutation::ARGB },

            vk::Format::A2B10G10R10_UNORM_PACK32 => Format { r: 10, g: 10, b: 10, a: 2, ty: FormatType::UNorm, permutation: Permutation::ABGR },
            vk::Format::A2B10G10R10_SNORM_PACK32 => Format { r: 10, g: 10, b: 10, a: 2, ty: FormatType::SNorm, permutation: Permutation::ABGR },
            vk::Format::A2B10G10R10_USCALED_PACK32 => Format { r: 10, g: 10, b: 10, a: 2, ty: FormatType::UScaled, permutation: Permutation::ABGR },
            vk::Format::A2B10G10R10_SSCALED_PACK32 => Format { r: 10, g: 10, b: 10, a: 2, ty: FormatType::SScaled, permutation: Permutation::ABGR },
            vk::Format::A2B10G10R10_UINT_PACK32 => Format { r: 10, g: 10, b: 10, a: 2, ty: FormatType::UInt, permutation: Permutation::ABGR },
            vk::Format::A2B10G10R10_SINT_PACK32 => Format { r: 10, g: 10, b: 10, a: 2, ty: FormatType::SInt, permutation: Permutation::ABGR },

            vk::Format::R16_UNORM => Format { r: 16, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::R },
            vk::Format::R16_SNORM => Format { r: 16, g: 0, b: 0, a: 0, ty: FormatType::SNorm, permutation: Permutation::R },
            vk::Format::R16_USCALED => Format { r: 16, g: 0, b: 0, a: 0, ty: FormatType::UScaled, permutation: Permutation::R },
            vk::Format::R16_SSCALED => Format { r: 16, g: 0, b: 0, a: 0, ty: FormatType::SScaled, permutation: Permutation::R },
            vk::Format::R16_UINT => Format { r: 16, g: 0, b: 0, a: 0, ty: FormatType::UInt, permutation: Permutation::R },
            vk::Format::R16_SINT => Format { r: 16, g: 0, b: 0, a: 0, ty: FormatType::SInt, permutation: Permutation::R },
            vk::Format::R16_SFLOAT => Format { r: 16, g: 0, b: 0, a: 0, ty: FormatType::SFloat, permutation: Permutation::R },

            vk::Format::R16G16_UNORM => Format { r: 16, g: 16, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::RG },
            vk::Format::R16G16_SNORM => Format { r: 16, g: 16, b: 0, a: 0, ty: FormatType::SNorm, permutation: Permutation::RG },
            vk::Format::R16G16_USCALED => Format { r: 16, g: 16, b: 0, a: 0, ty: FormatType::UScaled, permutation: Permutation::RG },
            vk::Format::R16G16_SSCALED => Format { r: 16, g: 16, b: 0, a: 0, ty: FormatType::SScaled, permutation: Permutation::RG },
            vk::Format::R16G16_UINT => Format { r: 16, g: 16, b: 0, a: 0, ty: FormatType::UInt, permutation: Permutation::RG },
            vk::Format::R16G16_SINT => Format { r: 16, g: 16, b: 0, a: 0, ty: FormatType::SInt, permutation: Permutation::RG },
            vk::Format::R16G16_SFLOAT => Format { r: 16, g: 16, b: 0, a: 0, ty: FormatType::SFloat, permutation: Permutation::RG },

            vk::Format::R16G16B16_UNORM => Format { r: 16, g: 16, b: 16, a: 0, ty: FormatType::UNorm, permutation: Permutation::RGB },
            vk::Format::R16G16B16_SNORM => Format { r: 16, g: 16, b: 16, a: 0, ty: FormatType::SNorm, permutation: Permutation::RGB },
            vk::Format::R16G16B16_USCALED => Format { r: 16, g: 16, b: 16, a: 0, ty: FormatType::UScaled, permutation: Permutation::RGB },
            vk::Format::R16G16B16_SSCALED => Format { r: 16, g: 16, b: 16, a: 0, ty: FormatType::SScaled, permutation: Permutation::RGB },
            vk::Format::R16G16B16_UINT => Format { r: 16, g: 16, b: 16, a: 0, ty: FormatType::UInt, permutation: Permutation::RGB },
            vk::Format::R16G16B16_SINT => Format { r: 16, g: 16, b: 16, a: 0, ty: FormatType::SInt, permutation: Permutation::RGB },
            vk::Format::R16G16B16_SFLOAT => Format { r: 16, g: 16, b: 16, a: 0, ty: FormatType::SFloat, permutation: Permutation::RGB },

            vk::Format::R16G16B16A16_UNORM => Format { r: 16, g: 16, b: 16, a: 16, ty: FormatType::UNorm, permutation: Permutation::RGBA },
            vk::Format::R16G16B16A16_SNORM => Format { r: 16, g: 16, b: 16, a: 16, ty: FormatType::SNorm, permutation: Permutation::RGBA },
            vk::Format::R16G16B16A16_USCALED => Format { r: 16, g: 16, b: 16, a: 16, ty: FormatType::UScaled, permutation: Permutation::RGBA },
            vk::Format::R16G16B16A16_SSCALED => Format { r: 16, g: 16, b: 16, a: 16, ty: FormatType::SScaled, permutation: Permutation::RGBA },
            vk::Format::R16G16B16A16_UINT => Format { r: 16, g: 16, b: 16, a: 16, ty: FormatType::UInt, permutation: Permutation::RGBA },
            vk::Format::R16G16B16A16_SINT => Format { r: 16, g: 16, b: 16, a: 16, ty: FormatType::SInt, permutation: Permutation::RGBA },
            vk::Format::R16G16B16A16_SFLOAT => Format { r: 16, g: 16, b: 16, a: 16, ty: FormatType::SFloat, permutation: Permutation::RGBA },

            vk::Format::R32_UINT => Format { r: 32, g: 0, b: 0, a: 0, ty: FormatType::UInt, permutation: Permutation::R },
            vk::Format::R32_SINT => Format { r: 32, g: 0, b: 0, a: 0, ty: FormatType::SInt, permutation: Permutation::R },
            vk::Format::R32_SFLOAT => Format { r: 32, g: 0, b: 0, a: 0, ty: FormatType::SFloat, permutation: Permutation::R },

            vk::Format::R32G32_UINT => Format { r: 32, g: 32, b: 0, a: 0, ty: FormatType::UInt, permutation: Permutation::RG },
            vk::Format::R32G32_SINT => Format { r: 32, g: 32, b: 0, a: 0, ty: FormatType::SInt, permutation: Permutation::RG },
            vk::Format::R32G32_SFLOAT => Format { r: 32, g: 32, b: 0, a: 0, ty: FormatType::SFloat, permutation: Permutation::RG },

            vk::Format::R32G32B32_UINT => Format { r: 32, g: 32, b: 32, a: 0, ty: FormatType::UInt, permutation: Permutation::RGB },
            vk::Format::R32G32B32_SINT => Format { r: 32, g: 32, b: 32, a: 0, ty: FormatType::SInt, permutation: Permutation::RGB },
            vk::Format::R32G32B32_SFLOAT => Format { r: 32, g: 32, b: 32, a: 0, ty: FormatType::SFloat, permutation: Permutation::RGB },

            vk::Format::R32G32B32A32_UINT => Format { r: 32, g: 32, b: 32, a: 32, ty: FormatType::UInt, permutation: Permutation::RGBA },
            vk::Format::R32G32B32A32_SINT => Format { r: 32, g: 32, b: 32, a: 32, ty: FormatType::SInt, permutation: Permutation::RGBA },
            vk::Format::R32G32B32A32_SFLOAT => Format { r: 32, g: 32, b: 32, a: 32, ty: FormatType::SFloat, permutation: Permutation::RGBA },

            vk::Format::R64_UINT => Format { r: 64, g: 0, b: 0, a: 0, ty: FormatType::UInt, permutation: Permutation::R },
            vk::Format::R64_SINT => Format { r: 64, g: 0, b: 0, a: 0, ty: FormatType::SInt, permutation: Permutation::R },
            vk::Format::R64_SFLOAT => Format { r: 64, g: 0, b: 0, a: 0, ty: FormatType::SFloat, permutation: Permutation::R },

            vk::Format::R64G64_UINT => Format { r: 64, g: 64, b: 0, a: 0, ty: FormatType::UInt, permutation: Permutation::RG },
            vk::Format::R64G64_SINT => Format { r: 64, g: 64, b: 0, a: 0, ty: FormatType::SInt, permutation: Permutation::RG },
            vk::Format::R64G64_SFLOAT => Format { r: 64, g: 64, b: 0, a: 0, ty: FormatType::SFloat, permutation: Permutation::RG },

            vk::Format::R64G64B64_UINT => Format { r: 64, g: 64, b: 64, a: 0, ty: FormatType::UInt, permutation: Permutation::RGB },
            vk::Format::R64G64B64_SINT => Format { r: 64, g: 64, b: 64, a: 0, ty: FormatType::SInt, permutation: Permutation::RGB },
            vk::Format::R64G64B64_SFLOAT => Format { r: 64, g: 64, b: 64, a: 0, ty: FormatType::SFloat, permutation: Permutation::RGB },

            vk::Format::R64G64B64A64_UINT => Format { r: 64, g: 64, b: 64, a: 64, ty: FormatType::UInt, permutation: Permutation::RGBA },
            vk::Format::R64G64B64A64_SINT => Format { r: 64, g: 64, b: 64, a: 64, ty: FormatType::SInt, permutation: Permutation::RGBA },
            vk::Format::R64G64B64A64_SFLOAT => Format { r: 64, g: 64, b: 64, a: 64, ty: FormatType::SFloat, permutation: Permutation::RGBA },

            vk::Format::B10G11R11_UFLOAT_PACK32 => Format { r: 11, g: 11, b: 10, a: 0, ty: FormatType::UFloat, permutation: Permutation::BGR },
            vk::Format::E5B9G9R9_UFLOAT_PACK32 => Format { r: 9, g: 9, b: 9, a: 5, ty: FormatType::UFloat, permutation: Permutation::EBGR },

            vk::Format::D16_UNORM => Format { r: 16, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::D },
            vk::Format::X8_D24_UNORM_PACK32 => Format { r: 24, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::D },
            vk::Format::D32_SFLOAT => Format { r: 32, g: 0, b: 0, a: 0, ty: FormatType::SFloat, permutation: Permutation::D },
            vk::Format::S8_UINT => Format { r: 8, g: 0, b: 0, a: 0, ty: FormatType::UInt, permutation: Permutation::S },

            vk::Format::D16_UNORM_S8_UINT => Format { r: 16, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::DS },
            vk::Format::D24_UNORM_S8_UINT => Format { r: 24, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::DS },
            vk::Format::D32_SFLOAT_S8_UINT => Format { r: 32, g: 0, b: 0, a: 0, ty: FormatType::SFloat, permutation: Permutation::DS },

            vk::Format::BC1_RGB_UNORM_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::BC1_RGB },
            vk::Format::BC1_RGB_SRGB_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::BC1_RGB },
            vk::Format::BC1_RGBA_UNORM_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::BC1_RGBA },
            vk::Format::BC1_RGBA_SRGB_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::BC1_RGBA },
            vk::Format::BC2_UNORM_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::BC2 },
            vk::Format::BC2_SRGB_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::BC2 },
            vk::Format::BC3_UNORM_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::BC3 },
            vk::Format::BC3_SRGB_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::BC3 },
            vk::Format::BC4_UNORM_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::BC4 },
            vk::Format::BC4_SNORM_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::SNorm, permutation: Permutation::BC4 },
            vk::Format::BC5_UNORM_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::BC5 },
            vk::Format::BC5_SNORM_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::SNorm, permutation: Permutation::BC5 },
            vk::Format::BC6H_UFLOAT_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::UFloat, permutation: Permutation::BC6H },
            vk::Format::BC6H_SFLOAT_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::SFloat, permutation: Permutation::BC6H },
            vk::Format::BC7_UNORM_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::BC7 },
            vk::Format::BC7_SRGB_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::BC7 },

            vk::Format::ETC2_R8G8B8_UNORM_BLOCK => Format { r: 8, g: 8, b: 8, a: 0, ty: FormatType::UNorm, permutation: Permutation::ETC2_RGB },
            vk::Format::ETC2_R8G8B8_SRGB_BLOCK => Format { r: 8, g: 8, b: 8, a: 0, ty: FormatType::sRGB, permutation: Permutation::ETC2_RGB },
            vk::Format::ETC2_R8G8B8A1_UNORM_BLOCK => Format { r: 8, g: 8, b: 8, a: 1, ty: FormatType::UNorm, permutation: Permutation::ETC2_RGBA },
            vk::Format::ETC2_R8G8B8A1_SRGB_BLOCK => Format { r: 8, g: 8, b: 8, a: 1, ty: FormatType::sRGB, permutation: Permutation::ETC2_RGBA },
            vk::Format::ETC2_R8G8B8A8_UNORM_BLOCK => Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::UNorm, permutation: Permutation::ETC2_RGBA },
            vk::Format::ETC2_R8G8B8A8_SRGB_BLOCK => Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::sRGB, permutation: Permutation::ETC2_RGBA },

            vk::Format::EAC_R11_UNORM_BLOCK => Format { r: 11, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::EAC_R },
            vk::Format::EAC_R11_SNORM_BLOCK => Format { r: 11, g: 0, b: 0, a: 0, ty: FormatType::SNorm, permutation: Permutation::EAC_R },
            vk::Format::EAC_R11G11_UNORM_BLOCK => Format { r: 11, g: 11, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::EAC_RG },
            vk::Format::EAC_R11G11_SNORM_BLOCK => Format { r: 11, g: 11, b: 0, a: 0, ty: FormatType::SNorm, permutation: Permutation::EAC_RG },

            vk::Format::ASTC_4X4_UNORM_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::ASTC { x: 4, y: 4 } },
            vk::Format::ASTC_4X4_SRGB_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::ASTC { x: 4, y: 4 } },
            vk::Format::ASTC_5X4_UNORM_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::ASTC { x: 5, y: 4 } },
            vk::Format::ASTC_5X4_SRGB_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::ASTC { x: 5, y: 4 } },
            vk::Format::ASTC_5X5_UNORM_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::ASTC { x: 5, y: 5 } },
            vk::Format::ASTC_5X5_SRGB_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::ASTC { x: 5, y: 5 } },
            vk::Format::ASTC_6X5_UNORM_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::ASTC { x: 6, y: 5 } },
            vk::Format::ASTC_6X5_SRGB_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::ASTC { x: 6, y: 5 } },
            vk::Format::ASTC_6X6_UNORM_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::ASTC { x: 6, y: 6 } },
            vk::Format::ASTC_6X6_SRGB_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::ASTC { x: 6, y: 6 } },
            vk::Format::ASTC_8X5_UNORM_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::ASTC { x: 8, y: 5 } },
            vk::Format::ASTC_8X5_SRGB_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::ASTC { x: 8, y: 5 } },
            vk::Format::ASTC_8X6_UNORM_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::ASTC { x: 8, y: 6 } },
            vk::Format::ASTC_8X6_SRGB_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::ASTC { x: 8, y: 6 } },
            vk::Format::ASTC_8X8_UNORM_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::ASTC { x: 8, y: 8 } },
            vk::Format::ASTC_8X8_SRGB_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::ASTC { x: 8, y: 8 } },
            vk::Format::ASTC_10X5_UNORM_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::ASTC { x: 10, y: 5 } },
            vk::Format::ASTC_10X5_SRGB_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::ASTC { x: 10, y: 5 } },
            vk::Format::ASTC_10X6_UNORM_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::ASTC { x: 10, y: 6 } },
            vk::Format::ASTC_10X6_SRGB_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::ASTC { x: 10, y: 6 } },
            vk::Format::ASTC_10X8_UNORM_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::ASTC { x: 10, y: 8 } },
            vk::Format::ASTC_10X8_SRGB_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::ASTC { x: 10, y: 8 } },
            vk::Format::ASTC_10X10_UNORM_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::ASTC { x: 10, y: 10 } },
            vk::Format::ASTC_10X10_SRGB_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::ASTC { x: 10, y: 10 } },
            vk::Format::ASTC_12X10_UNORM_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::ASTC { x: 12, y: 10 } },
            vk::Format::ASTC_12X10_SRGB_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::ASTC { x: 12, y: 10 } },
            vk::Format::ASTC_12X12_UNORM_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::ASTC { x: 12, y: 12 } },
            vk::Format::ASTC_12X12_SRGB_BLOCK => Format { r: 0, g: 0, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::ASTC { x: 12, y: 12 } },
            _ => panic!(),
        }
    }
}

impl TryFrom<Format> for vk::Format {
    type Error = Format;

    #[rustfmt::skip]
    fn try_from(value: Format) -> Result<Self, Self::Error> {
        let format = match value {
            Format { r: 4, g: 4, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::RG } => vk::Format::R4G4_UNORM_PACK8,
            Format { r: 4, g: 4, b: 4, a: 4, ty: FormatType::UNorm, permutation: Permutation::RGBA } => vk::Format::R4G4B4A4_UNORM_PACK16,
            Format { r: 4, g: 4, b: 4, a: 4, ty: FormatType::UNorm, permutation: Permutation::BGRA } => vk::Format::B4G4R4A4_UNORM_PACK16,
            Format { r: 5, g: 6, b: 5, a: 0, ty: FormatType::UNorm, permutation: Permutation::RGB } => vk::Format::R5G6B5_UNORM_PACK16,
            Format { r: 5, g: 6, b: 5, a: 0, ty: FormatType::UNorm, permutation: Permutation::BGR } => vk::Format::B5G6R5_UNORM_PACK16,
            Format { r: 5, g: 5, b: 5, a: 1, ty: FormatType::UNorm, permutation: Permutation::RGBA } => vk::Format::R5G5B5A1_UNORM_PACK16,
            Format { r: 5, g: 5, b: 5, a: 1, ty: FormatType::UNorm, permutation: Permutation::BGRA } => vk::Format::B5G5R5A1_UNORM_PACK16,
            Format { r: 5, g: 5, b: 5, a: 1, ty: FormatType::UNorm, permutation: Permutation::ARGB } => vk::Format::A1R5G5B5_UNORM_PACK16,
            Format { r: 8, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::R } => vk::Format::R8_UNORM,
            Format { r: 8, g: 0, b: 0, a: 0, ty: FormatType::SNorm, permutation: Permutation::R } => vk::Format::R8_SNORM,
            Format { r: 8, g: 0, b: 0, a: 0, ty: FormatType::UScaled, permutation: Permutation::R } => vk::Format::R8_USCALED,
            Format { r: 8, g: 0, b: 0, a: 0, ty: FormatType::SScaled, permutation: Permutation::R } => vk::Format::R8_SSCALED,
            Format { r: 8, g: 0, b: 0, a: 0, ty: FormatType::UInt, permutation: Permutation::R } => vk::Format::R8_UINT,
            Format { r: 8, g: 0, b: 0, a: 0, ty: FormatType::SInt, permutation: Permutation::R } => vk::Format::R8_SINT,
            Format { r: 8, g: 0, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::R } => vk::Format::R8_SRGB,
            Format { r: 8, g: 8, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::RG } => vk::Format::R8G8_UNORM,
            Format { r: 8, g: 8, b: 0, a: 0, ty: FormatType::SNorm, permutation: Permutation::RG } => vk::Format::R8G8_SNORM,
            Format { r: 8, g: 8, b: 0, a: 0, ty: FormatType::UScaled, permutation: Permutation::RG } => vk::Format::R8G8_USCALED,
            Format { r: 8, g: 8, b: 0, a: 0, ty: FormatType::SScaled, permutation: Permutation::RG } => vk::Format::R8G8_SSCALED,
            Format { r: 8, g: 8, b: 0, a: 0, ty: FormatType::UInt, permutation: Permutation::RG } => vk::Format::R8G8_UINT,
            Format { r: 8, g: 8, b: 0, a: 0, ty: FormatType::SInt, permutation: Permutation::RG } => vk::Format::R8G8_SINT,
            Format { r: 8, g: 8, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::RG } => vk::Format::R8G8_SRGB,
            Format { r: 8, g: 8, b: 8, a: 0, ty: FormatType::UNorm, permutation: Permutation::RGB } => vk::Format::R8G8B8_UNORM,
            Format { r: 8, g: 8, b: 8, a: 0, ty: FormatType::SNorm, permutation: Permutation::RGB } => vk::Format::R8G8B8_SNORM,
            Format { r: 8, g: 8, b: 8, a: 0, ty: FormatType::UScaled, permutation: Permutation::RGB } => vk::Format::R8G8B8_USCALED,
            Format { r: 8, g: 8, b: 8, a: 0, ty: FormatType::SScaled, permutation: Permutation::RGB } => vk::Format::R8G8B8_SSCALED,
            Format { r: 8, g: 8, b: 8, a: 0, ty: FormatType::UInt, permutation: Permutation::RGB } => vk::Format::R8G8B8_UINT,
            Format { r: 8, g: 8, b: 8, a: 0, ty: FormatType::SInt, permutation: Permutation::RGB } => vk::Format::R8G8B8_SINT,
            Format { r: 8, g: 8, b: 8, a: 0, ty: FormatType::sRGB, permutation: Permutation::RGB } => vk::Format::R8G8B8_SRGB,
            Format { r: 8, g: 8, b: 8, a: 0, ty: FormatType::UNorm, permutation: Permutation::BGR } => vk::Format::B8G8R8_UNORM,
            Format { r: 8, g: 8, b: 8, a: 0, ty: FormatType::SNorm, permutation: Permutation::BGR } => vk::Format::B8G8R8_SNORM,
            Format { r: 8, g: 8, b: 8, a: 0, ty: FormatType::UScaled, permutation: Permutation::BGR } => vk::Format::B8G8R8_USCALED,
            Format { r: 8, g: 8, b: 8, a: 0, ty: FormatType::SScaled, permutation: Permutation::BGR } => vk::Format::B8G8R8_SSCALED,
            Format { r: 8, g: 8, b: 8, a: 0, ty: FormatType::UInt, permutation: Permutation::BGR } => vk::Format::B8G8R8_UINT,
            Format { r: 8, g: 8, b: 8, a: 0, ty: FormatType::SInt, permutation: Permutation::BGR } => vk::Format::B8G8R8_SINT,
            Format { r: 8, g: 8, b: 8, a: 0, ty: FormatType::sRGB, permutation: Permutation::BGR } => vk::Format::B8G8R8_SRGB,
            Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::UNorm, permutation: Permutation::RGBA } => vk::Format::R8G8B8A8_UNORM,
            Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::SNorm, permutation: Permutation::RGBA } => vk::Format::R8G8B8A8_SNORM,
            Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::UScaled, permutation: Permutation::RGBA } => vk::Format::R8G8B8A8_USCALED,
            Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::SScaled, permutation: Permutation::RGBA } => vk::Format::R8G8B8A8_SSCALED,
            Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::UInt, permutation: Permutation::RGBA } => vk::Format::R8G8B8A8_UINT,
            Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::SInt, permutation: Permutation::RGBA } => vk::Format::R8G8B8A8_SINT,
            Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::sRGB, permutation: Permutation::RGBA } => vk::Format::R8G8B8A8_SRGB,
            Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::UNorm, permutation: Permutation::BGRA } => vk::Format::B8G8R8A8_UNORM,
            Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::SNorm, permutation: Permutation::BGRA } => vk::Format::B8G8R8A8_SNORM,
            Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::UScaled, permutation: Permutation::BGRA } => vk::Format::B8G8R8A8_USCALED,
            Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::SScaled, permutation: Permutation::BGRA } => vk::Format::B8G8R8A8_SSCALED,
            Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::UInt, permutation: Permutation::BGRA } => vk::Format::B8G8R8A8_UINT,
            Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::SInt, permutation: Permutation::BGRA } => vk::Format::B8G8R8A8_SINT,
            Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::sRGB, permutation: Permutation::BGRA } => vk::Format::B8G8R8A8_SRGB,
            Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::UNorm, permutation: Permutation::ABGR } => vk::Format::A8B8G8R8_UNORM_PACK32,
            Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::SNorm, permutation: Permutation::ABGR } => vk::Format::A8B8G8R8_SNORM_PACK32,
            Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::UScaled, permutation: Permutation::ABGR } => vk::Format::A8B8G8R8_USCALED_PACK32,
            Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::SScaled, permutation: Permutation::ABGR } => vk::Format::A8B8G8R8_SSCALED_PACK32,
            Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::UInt, permutation: Permutation::ABGR } => vk::Format::A8B8G8R8_UINT_PACK32,
            Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::SInt, permutation: Permutation::ABGR } => vk::Format::A8B8G8R8_SINT_PACK32,
            Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::sRGB, permutation: Permutation::ABGR } => vk::Format::A8B8G8R8_SRGB_PACK32,
            Format { r: 10, g: 10, b: 10, a: 2, ty: FormatType::UNorm, permutation: Permutation::ARGB } => vk::Format::A2R10G10B10_UNORM_PACK32,
            Format { r: 10, g: 10, b: 10, a: 2, ty: FormatType::SNorm, permutation: Permutation::ARGB } => vk::Format::A2R10G10B10_SNORM_PACK32,
            Format { r: 10, g: 10, b: 10, a: 2, ty: FormatType::UScaled, permutation: Permutation::ARGB } => vk::Format::A2R10G10B10_USCALED_PACK32,
            Format { r: 10, g: 10, b: 10, a: 2, ty: FormatType::SScaled, permutation: Permutation::ARGB } => vk::Format::A2R10G10B10_SSCALED_PACK32,
            Format { r: 10, g: 10, b: 10, a: 2, ty: FormatType::UInt, permutation: Permutation::ARGB } => vk::Format::A2R10G10B10_UINT_PACK32,
            Format { r: 10, g: 10, b: 10, a: 2, ty: FormatType::SInt, permutation: Permutation::ARGB } => vk::Format::A2R10G10B10_SINT_PACK32,
            Format { r: 10, g: 10, b: 10, a: 2, ty: FormatType::UNorm, permutation: Permutation::ABGR } => vk::Format::A2B10G10R10_UNORM_PACK32,
            Format { r: 10, g: 10, b: 10, a: 2, ty: FormatType::SNorm, permutation: Permutation::ABGR } => vk::Format::A2B10G10R10_SNORM_PACK32,
            Format { r: 10, g: 10, b: 10, a: 2, ty: FormatType::UScaled, permutation: Permutation::ABGR } => vk::Format::A2B10G10R10_USCALED_PACK32,
            Format { r: 10, g: 10, b: 10, a: 2, ty: FormatType::SScaled, permutation: Permutation::ABGR } => vk::Format::A2B10G10R10_SSCALED_PACK32,
            Format { r: 10, g: 10, b: 10, a: 2, ty: FormatType::UInt, permutation: Permutation::ABGR } => vk::Format::A2B10G10R10_UINT_PACK32,
            Format { r: 10, g: 10, b: 10, a: 2, ty: FormatType::SInt, permutation: Permutation::ABGR } => vk::Format::A2B10G10R10_SINT_PACK32,
            Format { r: 16, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::R } => vk::Format::R16_UNORM,
            Format { r: 16, g: 0, b: 0, a: 0, ty: FormatType::SNorm, permutation: Permutation::R } => vk::Format::R16_SNORM,
            Format { r: 16, g: 0, b: 0, a: 0, ty: FormatType::UScaled, permutation: Permutation::R } => vk::Format::R16_USCALED,
            Format { r: 16, g: 0, b: 0, a: 0, ty: FormatType::SScaled, permutation: Permutation::R } => vk::Format::R16_SSCALED,
            Format { r: 16, g: 0, b: 0, a: 0, ty: FormatType::UInt, permutation: Permutation::R } => vk::Format::R16_UINT,
            Format { r: 16, g: 0, b: 0, a: 0, ty: FormatType::SInt, permutation: Permutation::R } => vk::Format::R16_SINT,
            Format { r: 16, g: 0, b: 0, a: 0, ty: FormatType::SFloat, permutation: Permutation::R } => vk::Format::R16_SFLOAT,
            Format { r: 16, g: 16, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::RG } => vk::Format::R16G16_UNORM,
            Format { r: 16, g: 16, b: 0, a: 0, ty: FormatType::SNorm, permutation: Permutation::RG } => vk::Format::R16G16_SNORM,
            Format { r: 16, g: 16, b: 0, a: 0, ty: FormatType::UScaled, permutation: Permutation::RG } => vk::Format::R16G16_USCALED,
            Format { r: 16, g: 16, b: 0, a: 0, ty: FormatType::SScaled, permutation: Permutation::RG } => vk::Format::R16G16_SSCALED,
            Format { r: 16, g: 16, b: 0, a: 0, ty: FormatType::UInt, permutation: Permutation::RG } => vk::Format::R16G16_UINT,
            Format { r: 16, g: 16, b: 0, a: 0, ty: FormatType::SInt, permutation: Permutation::RG } => vk::Format::R16G16_SINT,
            Format { r: 16, g: 16, b: 0, a: 0, ty: FormatType::SFloat, permutation: Permutation::RG } => vk::Format::R16G16_SFLOAT,
            Format { r: 16, g: 16, b: 16, a: 0, ty: FormatType::UNorm, permutation: Permutation::RGB } => vk::Format::R16G16B16_UNORM,
            Format { r: 16, g: 16, b: 16, a: 0, ty: FormatType::SNorm, permutation: Permutation::RGB } => vk::Format::R16G16B16_SNORM,
            Format { r: 16, g: 16, b: 16, a: 0, ty: FormatType::UScaled, permutation: Permutation::RGB } => vk::Format::R16G16B16_USCALED,
            Format { r: 16, g: 16, b: 16, a: 0, ty: FormatType::SScaled, permutation: Permutation::RGB } => vk::Format::R16G16B16_SSCALED,
            Format { r: 16, g: 16, b: 16, a: 0, ty: FormatType::UInt, permutation: Permutation::RGB } => vk::Format::R16G16B16_UINT,
            Format { r: 16, g: 16, b: 16, a: 0, ty: FormatType::SInt, permutation: Permutation::RGB } => vk::Format::R16G16B16_SINT,
            Format { r: 16, g: 16, b: 16, a: 0, ty: FormatType::SFloat, permutation: Permutation::RGB } => vk::Format::R16G16B16_SFLOAT,
            Format { r: 16, g: 16, b: 16, a: 16, ty: FormatType::UNorm, permutation: Permutation::RGBA } => vk::Format::R16G16B16A16_UNORM,
            Format { r: 16, g: 16, b: 16, a: 16, ty: FormatType::SNorm, permutation: Permutation::RGBA } => vk::Format::R16G16B16A16_SNORM,
            Format { r: 16, g: 16, b: 16, a: 16, ty: FormatType::UScaled, permutation: Permutation::RGBA } => vk::Format::R16G16B16A16_USCALED,
            Format { r: 16, g: 16, b: 16, a: 16, ty: FormatType::SScaled, permutation: Permutation::RGBA } => vk::Format::R16G16B16A16_SSCALED,
            Format { r: 16, g: 16, b: 16, a: 16, ty: FormatType::UInt, permutation: Permutation::RGBA } => vk::Format::R16G16B16A16_UINT,
            Format { r: 16, g: 16, b: 16, a: 16, ty: FormatType::SInt, permutation: Permutation::RGBA } => vk::Format::R16G16B16A16_SINT,
            Format { r: 16, g: 16, b: 16, a: 16, ty: FormatType::SFloat, permutation: Permutation::RGBA } => vk::Format::R16G16B16A16_SFLOAT,
            Format { r: 32, g: 0, b: 0, a: 0, ty: FormatType::UInt, permutation: Permutation::R } => vk::Format::R32_UINT,
            Format { r: 32, g: 0, b: 0, a: 0, ty: FormatType::SInt, permutation: Permutation::R } => vk::Format::R32_SINT,
            Format { r: 32, g: 0, b: 0, a: 0, ty: FormatType::SFloat, permutation: Permutation::R } => vk::Format::R32_SFLOAT,
            Format { r: 32, g: 32, b: 0, a: 0, ty: FormatType::UInt, permutation: Permutation::RG } => vk::Format::R32G32_UINT,
            Format { r: 32, g: 32, b: 0, a: 0, ty: FormatType::SInt, permutation: Permutation::RG } => vk::Format::R32G32_SINT,
            Format { r: 32, g: 32, b: 0, a: 0, ty: FormatType::SFloat, permutation: Permutation::RG } => vk::Format::R32G32_SFLOAT,
            Format { r: 32, g: 32, b: 32, a: 0, ty: FormatType::UInt, permutation: Permutation::RGB } => vk::Format::R32G32B32_UINT,
            Format { r: 32, g: 32, b: 32, a: 0, ty: FormatType::SInt, permutation: Permutation::RGB } => vk::Format::R32G32B32_SINT,
            Format { r: 32, g: 32, b: 32, a: 0, ty: FormatType::SFloat, permutation: Permutation::RGB } => vk::Format::R32G32B32_SFLOAT,
            Format { r: 32, g: 32, b: 32, a: 32, ty: FormatType::UInt, permutation: Permutation::RGBA } => vk::Format::R32G32B32A32_UINT,
            Format { r: 32, g: 32, b: 32, a: 32, ty: FormatType::SInt, permutation: Permutation::RGBA } => vk::Format::R32G32B32A32_SINT,
            Format { r: 32, g: 32, b: 32, a: 32, ty: FormatType::SFloat, permutation: Permutation::RGBA } => vk::Format::R32G32B32A32_SFLOAT,
            Format { r: 64, g: 0, b: 0, a: 0, ty: FormatType::UInt, permutation: Permutation::R } => vk::Format::R64_UINT,
            Format { r: 64, g: 0, b: 0, a: 0, ty: FormatType::SInt, permutation: Permutation::R } => vk::Format::R64_SINT,
            Format { r: 64, g: 0, b: 0, a: 0, ty: FormatType::SFloat, permutation: Permutation::R } => vk::Format::R64_SFLOAT,
            Format { r: 64, g: 64, b: 0, a: 0, ty: FormatType::UInt, permutation: Permutation::RG } => vk::Format::R64G64_UINT,
            Format { r: 64, g: 64, b: 0, a: 0, ty: FormatType::SInt, permutation: Permutation::RG } => vk::Format::R64G64_SINT,
            Format { r: 64, g: 64, b: 0, a: 0, ty: FormatType::SFloat, permutation: Permutation::RG } => vk::Format::R64G64_SFLOAT,
            Format { r: 64, g: 64, b: 64, a: 0, ty: FormatType::UInt, permutation: Permutation::RGB } => vk::Format::R64G64B64_UINT,
            Format { r: 64, g: 64, b: 64, a: 0, ty: FormatType::SInt, permutation: Permutation::RGB } => vk::Format::R64G64B64_SINT,
            Format { r: 64, g: 64, b: 64, a: 0, ty: FormatType::SFloat, permutation: Permutation::RGB } => vk::Format::R64G64B64_SFLOAT,
            Format { r: 64, g: 64, b: 64, a: 64, ty: FormatType::UInt, permutation: Permutation::RGBA } => vk::Format::R64G64B64A64_UINT,
            Format { r: 64, g: 64, b: 64, a: 64, ty: FormatType::SInt, permutation: Permutation::RGBA } => vk::Format::R64G64B64A64_SINT,
            Format { r: 64, g: 64, b: 64, a: 64, ty: FormatType::SFloat, permutation: Permutation::RGBA } => vk::Format::R64G64B64A64_SFLOAT,
            Format { r: 11, g: 11, b: 10, a: 0, ty: FormatType::UFloat, permutation: Permutation::BGR } => vk::Format::B10G11R11_UFLOAT_PACK32,
            Format { r: 9, g: 9, b: 9, a: 5, ty: FormatType::UFloat, permutation: Permutation::EBGR } => vk::Format::E5B9G9R9_UFLOAT_PACK32,
            Format { r: 16, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::D } => vk::Format::D16_UNORM,
            Format { r: 24, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::D } => vk::Format::X8_D24_UNORM_PACK32,
            Format { r: 32, g: 0, b: 0, a: 0, ty: FormatType::SFloat, permutation: Permutation::D } => vk::Format::D32_SFLOAT,
            Format { r: 8, g: 0, b: 0, a: 0, ty: FormatType::UInt, permutation: Permutation::S } => vk::Format::S8_UINT,
            Format { r: 16, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::DS } => vk::Format::D16_UNORM_S8_UINT,
            Format { r: 24, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::DS } => vk::Format::D24_UNORM_S8_UINT,
            Format { r: 32, g: 0, b: 0, a: 0, ty: FormatType::SFloat, permutation: Permutation::DS } => vk::Format::D32_SFLOAT_S8_UINT,



            Format { ty: FormatType::UNorm, permutation: Permutation::BC1_RGB, .. } => vk::Format::BC1_RGB_UNORM_BLOCK,
            Format { ty: FormatType::sRGB, permutation: Permutation::BC1_RGB, .. } => vk::Format::BC1_RGB_SRGB_BLOCK,
            Format { ty: FormatType::UNorm, permutation: Permutation::BC1_RGBA, .. } => vk::Format::BC1_RGBA_UNORM_BLOCK,
            Format { ty: FormatType::sRGB, permutation: Permutation::BC1_RGBA, .. } => vk::Format::BC1_RGBA_SRGB_BLOCK,
            Format { ty: FormatType::UNorm, permutation: Permutation::BC2, .. } => vk::Format::BC2_UNORM_BLOCK,
            Format { ty: FormatType::sRGB, permutation: Permutation::BC2, .. } => vk::Format::BC2_SRGB_BLOCK,
            Format { ty: FormatType::UNorm, permutation: Permutation::BC3, .. } => vk::Format::BC3_UNORM_BLOCK,
            Format { ty: FormatType::sRGB, permutation: Permutation::BC3, .. } => vk::Format::BC3_SRGB_BLOCK,
            Format { ty: FormatType::UNorm, permutation: Permutation::BC4, .. } => vk::Format::BC4_UNORM_BLOCK,
            Format { ty: FormatType::SNorm, permutation: Permutation::BC4, .. } => vk::Format::BC4_SNORM_BLOCK,
            Format { ty: FormatType::UNorm, permutation: Permutation::BC5, .. } => vk::Format::BC5_UNORM_BLOCK,
            Format { ty: FormatType::SNorm, permutation: Permutation::BC5, .. } => vk::Format::BC5_SNORM_BLOCK,
            Format { ty: FormatType::UFloat, permutation: Permutation::BC6H, .. } => vk::Format::BC6H_UFLOAT_BLOCK,
            Format { ty: FormatType::SFloat, permutation: Permutation::BC6H, .. } => vk::Format::BC6H_SFLOAT_BLOCK,
            Format { ty: FormatType::UNorm, permutation: Permutation::BC7, .. } => vk::Format::BC7_UNORM_BLOCK,
            Format { ty: FormatType::sRGB, permutation: Permutation::BC7, .. } => vk::Format::BC7_SRGB_BLOCK,
            Format { r: 8, g: 8, b: 8, a: 0, ty: FormatType::UNorm, permutation: Permutation::ETC2_RGB } => vk::Format::ETC2_R8G8B8_UNORM_BLOCK,
            Format { r: 8, g: 8, b: 8, a: 0, ty: FormatType::sRGB, permutation: Permutation::ETC2_RGB } => vk::Format::ETC2_R8G8B8_SRGB_BLOCK,
            Format { r: 8, g: 8, b: 8, a: 1, ty: FormatType::UNorm, permutation: Permutation::ETC2_RGBA } => vk::Format::ETC2_R8G8B8A1_UNORM_BLOCK,
            Format { r: 8, g: 8, b: 8, a: 1, ty: FormatType::sRGB, permutation: Permutation::ETC2_RGBA } => vk::Format::ETC2_R8G8B8A1_SRGB_BLOCK,
            Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::UNorm, permutation: Permutation::ETC2_RGBA } => vk::Format::ETC2_R8G8B8A8_UNORM_BLOCK,
            Format { r: 8, g: 8, b: 8, a: 8, ty: FormatType::sRGB, permutation: Permutation::ETC2_RGBA } => vk::Format::ETC2_R8G8B8A8_SRGB_BLOCK,
            Format { r: 11, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::EAC_R } => vk::Format::EAC_R11_UNORM_BLOCK,
            Format { r: 11, g: 0, b: 0, a: 0, ty: FormatType::SNorm, permutation: Permutation::EAC_R } => vk::Format::EAC_R11_SNORM_BLOCK,
            Format { r: 11, g: 11, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::EAC_RG } => vk::Format::EAC_R11G11_UNORM_BLOCK,
            Format { r: 11, g: 11, b: 0, a: 0, ty: FormatType::SNorm, permutation: Permutation::EAC_RG } => vk::Format::EAC_R11G11_SNORM_BLOCK,
            Format { ty: FormatType::UNorm, permutation: Permutation::ASTC { x: 4, y: 4 }, ..} => vk::Format::ASTC_4X4_UNORM_BLOCK,
            Format { ty: FormatType::sRGB, permutation: Permutation::ASTC { x: 4, y: 4 }, ..} => vk::Format::ASTC_4X4_SRGB_BLOCK,
            Format { ty: FormatType::UNorm, permutation: Permutation::ASTC { x: 5, y: 4 }, ..} => vk::Format::ASTC_5X4_UNORM_BLOCK,
            Format { ty: FormatType::sRGB, permutation: Permutation::ASTC { x: 5, y: 4 }, ..} => vk::Format::ASTC_5X4_SRGB_BLOCK,
            Format { ty: FormatType::UNorm, permutation: Permutation::ASTC { x: 5, y: 5 }, ..} => vk::Format::ASTC_5X5_UNORM_BLOCK,
            Format { ty: FormatType::sRGB, permutation: Permutation::ASTC { x: 5, y: 5 }, ..} => vk::Format::ASTC_5X5_SRGB_BLOCK,
            Format { ty: FormatType::UNorm, permutation: Permutation::ASTC { x: 6, y: 5 }, ..} => vk::Format::ASTC_6X5_UNORM_BLOCK,
            Format { ty: FormatType::sRGB, permutation: Permutation::ASTC { x: 6, y: 5 }, ..} => vk::Format::ASTC_6X5_SRGB_BLOCK,
            Format { ty: FormatType::UNorm, permutation: Permutation::ASTC { x: 6, y: 6 }, ..} => vk::Format::ASTC_6X6_UNORM_BLOCK,
            Format { ty: FormatType::sRGB, permutation: Permutation::ASTC { x: 6, y: 6 }, ..} => vk::Format::ASTC_6X6_SRGB_BLOCK,
            Format { ty: FormatType::UNorm, permutation: Permutation::ASTC { x: 8, y: 5 }, ..} => vk::Format::ASTC_8X5_UNORM_BLOCK,
            Format { ty: FormatType::sRGB, permutation: Permutation::ASTC { x: 8, y: 5 }, ..} => vk::Format::ASTC_8X5_SRGB_BLOCK,
            Format { ty: FormatType::UNorm, permutation: Permutation::ASTC { x: 8, y: 6 }, ..} => vk::Format::ASTC_8X6_UNORM_BLOCK,
            Format { ty: FormatType::sRGB, permutation: Permutation::ASTC { x: 8, y: 6 }, ..} => vk::Format::ASTC_8X6_SRGB_BLOCK,
            Format { ty: FormatType::UNorm, permutation: Permutation::ASTC { x: 8, y: 8 }, ..} => vk::Format::ASTC_8X8_UNORM_BLOCK,
            Format { ty: FormatType::sRGB, permutation: Permutation::ASTC { x: 8, y: 8 }, ..} => vk::Format::ASTC_8X8_SRGB_BLOCK,
            Format { ty: FormatType::UNorm, permutation: Permutation::ASTC { x: 10, y: 5 }, ..} => vk::Format::ASTC_10X5_UNORM_BLOCK,
            Format { ty: FormatType::sRGB, permutation: Permutation::ASTC { x: 10, y: 5 }, ..} => vk::Format::ASTC_10X5_SRGB_BLOCK,
            Format { ty: FormatType::UNorm, permutation: Permutation::ASTC { x: 10, y: 6 }, ..} => vk::Format::ASTC_10X6_UNORM_BLOCK,
            Format { ty: FormatType::sRGB, permutation: Permutation::ASTC { x: 10, y: 6 }, ..} => vk::Format::ASTC_10X6_SRGB_BLOCK,
            Format { ty: FormatType::UNorm, permutation: Permutation::ASTC { x: 10, y: 8 }, ..} => vk::Format::ASTC_10X8_UNORM_BLOCK,
            Format { ty: FormatType::sRGB, permutation: Permutation::ASTC { x: 10, y: 8 }, ..} => vk::Format::ASTC_10X8_SRGB_BLOCK,
            Format { ty: FormatType::UNorm, permutation: Permutation::ASTC { x: 10, y: 10 }, ..} => vk::Format::ASTC_10X10_UNORM_BLOCK,
            Format { ty: FormatType::sRGB, permutation: Permutation::ASTC { x: 10, y: 10 }, ..} => vk::Format::ASTC_10X10_SRGB_BLOCK,
            Format { ty: FormatType::UNorm, permutation: Permutation::ASTC { x: 12, y: 10 }, ..} => vk::Format::ASTC_12X10_UNORM_BLOCK,
            Format { ty: FormatType::sRGB, permutation: Permutation::ASTC { x: 12, y: 10 }, ..} => vk::Format::ASTC_12X10_SRGB_BLOCK,
            Format { ty: FormatType::UNorm, permutation: Permutation::ASTC { x: 12, y: 12 }, ..} => vk::Format::ASTC_12X12_UNORM_BLOCK,
            Format { ty: FormatType::sRGB, permutation: Permutation::ASTC { x: 12, y: 12 }, ..} => vk::Format::ASTC_12X12_SRGB_BLOCK,
            _ => return Err(value),
        };
        Ok(format)
    }
}

#[derive(Clone, Debug)]
pub struct ColorSpace {
    pub primaries: ColorSpacePrimaries,
    pub transfer_function: ColorSpaceTransferFunction,
}

impl From<vk::ColorSpaceKHR> for ColorSpace {
    fn from(value: vk::ColorSpaceKHR) -> Self {
        match value {
            vk::ColorSpaceKHR::SRGB_NONLINEAR => ColorSpace {
                primaries: ColorSpacePrimaries::BT709,
                transfer_function: ColorSpaceTransferFunction::LINEAR,
            },
            vk::ColorSpaceKHR::DISPLAY_P3_NONLINEAR_EXT => ColorSpace {
                primaries: ColorSpacePrimaries::DCI_P3,
                transfer_function: ColorSpaceTransferFunction::DisplayP3,
            },
            vk::ColorSpaceKHR::DISPLAY_P3_LINEAR_EXT => ColorSpace {
                primaries: ColorSpacePrimaries::DCI_P3,
                transfer_function: ColorSpaceTransferFunction::LINEAR,
            },
            vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT => ColorSpace {
                primaries: ColorSpacePrimaries::BT709,
                transfer_function: ColorSpaceTransferFunction::LINEAR,
            },
            vk::ColorSpaceKHR::EXTENDED_SRGB_NONLINEAR_EXT => ColorSpace {
                primaries: ColorSpacePrimaries::BT709,
                transfer_function: ColorSpaceTransferFunction::scRGB,
            },
            vk::ColorSpaceKHR::DCI_P3_NONLINEAR_EXT => ColorSpace {
                primaries: ColorSpacePrimaries::XYZ,
                transfer_function: ColorSpaceTransferFunction::DCI_P3,
            },
            vk::ColorSpaceKHR::BT709_LINEAR_EXT => ColorSpace {
                primaries: ColorSpacePrimaries::BT709,
                transfer_function: ColorSpaceTransferFunction::LINEAR,
            },
            vk::ColorSpaceKHR::BT709_NONLINEAR_EXT => ColorSpace {
                primaries: ColorSpacePrimaries::BT709,
                transfer_function: ColorSpaceTransferFunction::ITU,
            },
            vk::ColorSpaceKHR::BT2020_LINEAR_EXT => ColorSpace {
                primaries: ColorSpacePrimaries::BT2020,
                transfer_function: ColorSpaceTransferFunction::LINEAR,
            },
            vk::ColorSpaceKHR::HDR10_ST2084_EXT => ColorSpace {
                primaries: ColorSpacePrimaries::BT2020,
                transfer_function: ColorSpaceTransferFunction::ST2084_PQ,
            },
            vk::ColorSpaceKHR::DOLBYVISION_EXT => ColorSpace {
                primaries: ColorSpacePrimaries::BT2020,
                transfer_function: ColorSpaceTransferFunction::ST2084_PQ,
            },
            vk::ColorSpaceKHR::HDR10_HLG_EXT => ColorSpace {
                primaries: ColorSpacePrimaries::BT2020,
                transfer_function: ColorSpaceTransferFunction::HLG,
            },
            vk::ColorSpaceKHR::ADOBERGB_LINEAR_EXT => ColorSpace {
                primaries: ColorSpacePrimaries::ADOBE_RGB,
                transfer_function: ColorSpaceTransferFunction::LINEAR,
            },
            vk::ColorSpaceKHR::ADOBERGB_NONLINEAR_EXT => ColorSpace {
                primaries: ColorSpacePrimaries::ADOBE_RGB,
                transfer_function: ColorSpaceTransferFunction::AdobeRGB,
            },
            _ => panic!(),
        }
    }
}
#[derive(Clone, Debug, PartialEq)]
pub struct ColorSpacePrimaries {
    pub r: glam::Vec2,
    pub g: glam::Vec2,
    pub b: glam::Vec2,
    pub white_point: glam::Vec2,
}

pub mod white_points {
    use glam::Vec2;
    pub const D65: Vec2 = Vec2::new(0.3127, 0.3290);
    pub const D60: Vec2 = Vec2::new(0.32168, 0.33767);
    pub const E: Vec2 = Vec2::new(0.3333, 0.3333);
}
impl ColorSpacePrimaries {
    /// Primaries for CIE 1931 XYZ colorspace.
    /// It is the most common color space for LDR content.
    /// https://en.wikipedia.org/wiki/Rec._709
    pub const BT709: Self = ColorSpacePrimaries {
        r: Vec2::new(0.64, 0.33),
        g: Vec2::new(0.3, 0.6),
        b: Vec2::new(0.15, 0.06),
        white_point: white_points::D65,
    };

    /// Primaries for CIE 1931 XYZ colorspace.
    /// Commonly used as a bridge color space between other color spaces.
    /// https://en.wikipedia.org/wiki/CIE_1931_color_space
    pub const XYZ: Self = ColorSpacePrimaries {
        r: Vec2::new(1.0, 0.0),
        g: Vec2::new(0.0, 1.0),
        b: Vec2::new(0.0, 0.0),
        white_point: white_points::E,
    };
    pub const DCI_P3: Self = ColorSpacePrimaries {
        r: Vec2::new(0.68, 0.32),
        g: Vec2::new(0.265, 0.69),
        b: Vec2::new(0.15, 0.06),
        white_point: white_points::D65,
    };
    pub const BT2020: Self = ColorSpacePrimaries {
        r: Vec2::new(0.708, 0.292),
        g: Vec2::new(0.170, 0.797),
        b: Vec2::new(0.131, 0.046),
        white_point: white_points::D65,
    };
    pub const ADOBE_RGB: Self = ColorSpacePrimaries {
        r: Vec2::new(0.64, 0.33),
        g: Vec2::new(0.21, 0.71),
        b: Vec2::new(0.15, 0.06),
        white_point: white_points::D65,
    };

    /// Primaries used in ACES2065-1.
    /// Typically, this is the colorspace you would use to transfer images/animations between production studios.
    pub const ACES_AP0: Self = ColorSpacePrimaries {
        r: Vec2::new(0.7347, 0.2653),
        g: Vec2::new(0.0, 1.0),
        b: Vec2::new(0.0001, -0.0770),
        white_point: white_points::D60,
    };

    /// Primaries used in ACEScg.
    /// ACEScg is the recommended rendering color space for HDR games.
    pub const ACES_AP1: Self = ColorSpacePrimaries {
        r: Vec2::new(0.713, 0.293),
        g: Vec2::new(0.165, 0.830),
        b: Vec2::new(0.128, 0.044),
        white_point: white_points::D60,
    };

    pub fn area_size(&self) -> f32 {
        let a = (self.r - self.g).length();
        let b = (self.g - self.b).length();
        let c = (self.b - self.r).length();
        let s = (a + b + c) / 2.0;
        let area = (s * (s - a) * (s - b) * (s - c)).sqrt();
        area
    }
    #[allow(non_snake_case)]
    pub fn to_xyz(&self) -> glam::Mat3 {
        use glam::{Mat3, Vec3, Vec3A, Vec4, Vec4Swizzles};
        let x = Vec4::new(self.r.x, self.g.x, self.b.x, self.white_point.x);
        let y = Vec4::new(self.r.y, self.g.y, self.b.y, self.white_point.y);
        let X = x / y;
        let Z = (1.0 - x - y) / y;

        let mat = Mat3::from_cols(X.xyz(), Vec3::ONE, Z.xyz()).transpose();
        let white_point = Vec3A::new(X.w, 1.0, Z.w);

        let S = mat.inverse() * white_point;
        mat * Mat3::from_diagonal(S.into())
    }

    pub fn to_color_space(&self, other_color_space: &Self) -> glam::Mat3 {
        if self == other_color_space {
            return glam::Mat3::IDENTITY;
        }
        if self == &ColorSpacePrimaries::XYZ {
            return other_color_space.to_xyz().inverse();
        }
        if other_color_space == &ColorSpacePrimaries::XYZ {
            return self.to_xyz();
        }
        other_color_space.to_xyz().inverse() * self.to_xyz()
    }
}

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorSpaceTransferFunction {
    LINEAR = 0,
    sRGB = 1,
    scRGB = 2,
    DCI_P3 = 3,
    DisplayP3 = 4,
    ITU = 5,
    ST2084_PQ = 6,
    HLG = 7,
    AdobeRGB = 8,
}

impl ColorSpaceTransferFunction {
    pub fn is_linear(&self) -> bool {
        matches!(self, Self::LINEAR)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_color_space_conversion() {
        let mat = super::ColorSpacePrimaries::ACES_AP1.to_xyz();
        println!("{:?}", mat);
    }
}
