//! Saving the `screenshot` command's screenshots: SDR ones as PNG, through the
//! platform's image encoder (ImageIO on macOS, WIC on Windows), and the others
//! as KTX2, which keeps their texels as they are, in their own format and color
//! space.

use std::fs::File;
use std::io::{BufWriter, Write as _};
use std::num::NonZeroU8;
use std::path::Path;

use bevy_pumicite::pumicite::ash::vk;
use dust_pbr::screenshot::Screenshot;
use ktx2::{
    ChannelTypeQualifiers, ColorModel, ColorPrimaries, DataFormatFlags, DfdBlockHeaderBasic,
    DfdHeader, Format, Header, Index, LevelIndex, SampleInformation, TransferFunction,
};

/// Saves `screenshot`, an SDR one in 8-bit sRGB RGBA, as a PNG at `path`, an
/// absolute path. The alpha channel is left out.
#[cfg(target_vendor = "apple")]
pub(super) fn save_png(path: &Path, screenshot: &Screenshot) -> Result<(), String> {
    use objc2_core_foundation::{CFData, CFString, CFURL};
    use objc2_core_graphics::{
        CGBitmapInfo, CGColorRenderingIntent, CGColorSpace, CGDataProvider, CGImage,
        CGImageAlphaInfo, kCGColorSpaceSRGB,
    };
    use objc2_image_io::CGImageDestination;

    if screenshot.format != vk::Format::R8G8B8A8_UNORM {
        return Err(format!(
            "error: can't save {:?} texels as PNG",
            screenshot.format
        ));
    }
    let (width, height) = (screenshot.extent.x as usize, screenshot.extent.y as usize);
    let provider = CGDataProvider::with_cf_data(Some(&CFData::from_bytes(screenshot.texels())))
        .ok_or("error: ImageIO can't take the screenshot's texels")?;
    let color_space = CGColorSpace::with_name(Some(unsafe { kCGColorSpaceSRGB }));
    // SAFETY: `decode` may be null, for no remapping of the values; the
    // provider holds `height` rows of `width` texels of 4 bytes.
    let image = unsafe {
        CGImage::new(
            width,
            height,
            8,
            32,
            width * 4,
            color_space.as_deref(),
            CGBitmapInfo(CGImageAlphaInfo::NoneSkipLast.0),
            Some(&provider),
            std::ptr::null(),
            false,
            CGColorRenderingIntent::RenderingIntentDefault,
        )
    }
    .ok_or("error: ImageIO can't make an image of the screenshot")?;
    let url = CFURL::from_file_path(path).ok_or("error: ImageIO can't take the path")?;
    // SAFETY: there are no options or properties, whose types would matter.
    unsafe {
        let destination =
            CGImageDestination::with_url(&url, &CFString::from_static_str("public.png"), 1, None)
                .ok_or("error: ImageIO can't write a PNG there")?;
        destination.add_image(&image, None);
        if !destination.finalize() {
            return Err("error: ImageIO failed to write the PNG".to_owned());
        }
    }
    Ok(())
}

/// Saves `screenshot`, an SDR one in 8-bit sRGB RGBA, as a PNG at `path`, an
/// absolute path. The alpha channel is left out.
#[cfg(windows)]
pub(super) fn save_png(path: &Path, screenshot: &Screenshot) -> Result<(), String> {
    use windows::Win32::Foundation::GENERIC_WRITE;
    use windows::Win32::Graphics::Imaging::{
        CLSID_WICImagingFactory, GUID_ContainerFormatPng, GUID_WICPixelFormat24bppBGR,
        IWICImagingFactory, WICBitmapEncoderNoCache,
    };
    use windows::Win32::System::Com::{
        CLSCTX_INPROC_SERVER, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx,
    };
    use windows::core::HSTRING;

    if screenshot.format != vk::Format::R8G8B8A8_UNORM {
        return Err(format!(
            "error: can't save {:?} texels as PNG",
            screenshot.format
        ));
    }
    let (width, height) = (screenshot.extent.x, screenshot.extent.y);
    // WIC's PNG encoder takes BGR, without the alpha channel.
    let bgr: Vec<u8> = screenshot
        .texels()
        .chunks_exact(4)
        .flat_map(|texel| [texel[2], texel[1], texel[0]])
        .collect();
    let wic = |error: windows::core::Error| format!("error: WIC failed to write the PNG: {error}");
    // SAFETY: COM is initialized on this thread before WIC is used, and the
    // calls get valid pointers: to the GUIDs, and to locals for the frame.
    unsafe {
        // Fails harmlessly if the thread already uses COM.
        let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
        let factory: IWICImagingFactory =
            CoCreateInstance(&CLSID_WICImagingFactory, None, CLSCTX_INPROC_SERVER).map_err(wic)?;
        let stream = factory.CreateStream().map_err(wic)?;
        stream
            .InitializeFromFilename(&HSTRING::from(path), GENERIC_WRITE.0)
            .map_err(wic)?;
        let encoder = factory
            .CreateEncoder(&GUID_ContainerFormatPng, std::ptr::null())
            .map_err(wic)?;
        encoder
            .Initialize(&stream, WICBitmapEncoderNoCache)
            .map_err(wic)?;
        let mut frame = None;
        let mut options = None;
        encoder
            .CreateNewFrame(&mut frame, &mut options)
            .map_err(wic)?;
        let frame = frame.ok_or("error: WIC made no frame for the PNG")?;
        frame.Initialize(options.as_ref()).map_err(wic)?;
        frame.SetSize(width, height).map_err(wic)?;
        // The encoder changes the format to one it supports, if it doesn't
        // support this one.
        let mut format = GUID_WICPixelFormat24bppBGR;
        frame.SetPixelFormat(&mut format).map_err(wic)?;
        if format != GUID_WICPixelFormat24bppBGR {
            return Err("error: WIC's PNG encoder doesn't take 24-bit BGR".to_owned());
        }
        frame.WritePixels(height, width * 3, &bgr).map_err(wic)?;
        frame.Commit().map_err(wic)?;
        encoder.Commit().map_err(wic)?;
    }
    Ok(())
}

/// Saves `screenshot` as KTX2 at `path`: its texels as they are, in their
/// own format and color space, without supercompression, after a data format
/// descriptor giving their channels, primaries and transfer function.
pub(super) fn save_ktx2(path: &Path, screenshot: &Screenshot) -> Result<(), String> {
    const ALPHA: u8 = 15;
    // Each channel's ID, bit offset and length; whether they're floats; and
    // the size of their type, which the KTX2 header gives.
    let (channels, float, type_size): (&[(u8, u16, u8)], bool, u32) = match screenshot.format {
        vk::Format::A2B10G10R10_UNORM_PACK32 => (
            &[(0, 0, 10), (1, 10, 10), (2, 20, 10), (ALPHA, 30, 2)],
            false,
            4,
        ),
        vk::Format::R16G16B16A16_SFLOAT => (
            &[(0, 0, 16), (1, 16, 16), (2, 32, 16), (ALPHA, 48, 16)],
            true,
            2,
        ),
        format => return Err(format!("error: can't save {format:?} texels as KTX2")),
    };
    let (primaries, transfer) = match screenshot.color_space {
        vk::ColorSpaceKHR::SRGB_NONLINEAR => (ColorPrimaries::BT709, TransferFunction::SRGB),
        vk::ColorSpaceKHR::BT709_LINEAR_EXT => (ColorPrimaries::BT709, TransferFunction::Linear),
        vk::ColorSpaceKHR::HDR10_ST2084_EXT => (ColorPrimaries::BT2020, TransferFunction::PQEOTF),
        color_space => return Err(format!("error: can't save {color_space:?} as KTX2")),
    };
    let texel_size: u16 = channels
        .iter()
        .map(|&(_, _, bits)| u16::from(bits))
        .sum::<u16>()
        / 8;

    let block_size = DfdHeader::LENGTH
        + DfdBlockHeaderBasic::LENGTH
        + SampleInformation::LENGTH * channels.len();
    // The descriptor starts with its own size.
    let mut dfd = ((4 + block_size) as u32).to_le_bytes().to_vec();
    dfd.extend(DfdHeader::BASIC.as_bytes(block_size as u16));
    dfd.extend(
        DfdBlockHeaderBasic {
            color_model: Some(ColorModel::RGBSDA),
            color_primaries: Some(primaries),
            transfer_function: Some(transfer),
            flags: DataFormatFlags::STRAIGHT_ALPHA,
            texel_block_dimensions: [NonZeroU8::MIN; 4],
            bytes_planes: [texel_size as u8, 0, 0, 0, 0, 0, 0, 0],
        }
        .as_bytes(),
    );
    for &(channel, offset, bits) in channels {
        let (qualifiers, lower, upper) = if float {
            (
                ChannelTypeQualifiers::FLOAT | ChannelTypeQualifiers::SIGNED,
                (-1.0f32).to_bits(),
                1.0f32.to_bits(),
            )
        } else {
            // The transfer function is for the color, not the alpha.
            let linear = if channel == ALPHA && transfer != TransferFunction::Linear {
                ChannelTypeQualifiers::LINEAR
            } else {
                ChannelTypeQualifiers::empty()
            };
            (linear, 0, (1 << bits) - 1)
        };
        dfd.extend(
            SampleInformation {
                bit_offset: offset,
                bit_length: NonZeroU8::new(bits).unwrap(),
                channel_type: channel,
                channel_type_qualifiers: qualifiers,
                sample_positions: [0; 4],
                lower,
                upper,
            }
            .as_bytes(),
        );
    }

    let texels = screenshot.texels();
    let dfd_offset = Header::LENGTH + LevelIndex::LENGTH;
    // The image starts at a multiple of the texel size, which is a multiple of
    // 4, as KTX2 asks.
    let padding = (dfd_offset + dfd.len()).next_multiple_of(usize::from(texel_size))
        - (dfd_offset + dfd.len());
    let header = Header {
        format: Format::new(screenshot.format.as_raw() as u32),
        type_size,
        pixel_width: screenshot.extent.x,
        pixel_height: screenshot.extent.y,
        pixel_depth: 0,
        layer_count: 0,
        face_count: 1,
        level_count: 1,
        supercompression_scheme: None,
        index: Index {
            dfd_byte_offset: dfd_offset as u32,
            dfd_byte_length: dfd.len() as u32,
            kvd_byte_offset: 0,
            kvd_byte_length: 0,
            sgd_byte_offset: 0,
            sgd_byte_length: 0,
        },
    };
    let level = LevelIndex {
        byte_offset: (dfd_offset + dfd.len() + padding) as u64,
        byte_length: texels.len() as u64,
        uncompressed_byte_length: texels.len() as u64,
    };
    let write = || {
        let mut file = BufWriter::new(File::create(path)?);
        file.write_all(&header.as_bytes())?;
        file.write_all(&level.as_bytes())?;
        file.write_all(&dfd)?;
        file.write_all(&vec![0; padding])?;
        file.write_all(texels)?;
        file.flush()
    };
    write().map_err(|error| format!("error: can't write {}: {error}", path.display()))
}
