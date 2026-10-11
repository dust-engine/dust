//! Closest-depth hierarchical Z buffer, rebuilt every frame from the depth
//! G-buffer by `hzb_build.slang` (one dispatch for the whole mip chain).

use bevy::prelude::*;
use bevy_pumicite::prelude::ComputePipeline;
use bevy_pumicite::staging::UniformRingBuffer;
use bevy_pumicite::SubmissionState;
use bytemuck::{Pod, Zeroable};
use dust_gfxdebug::{GpuProfiler, GpuTimerCommands};
use pumicite::ash::vk;
use pumicite::buffer::Buffer;
use pumicite::debug::DebugObject;
use pumicite::image::{FullImageView, Image, ImageExt, MipImageViews};
use pumicite::sync::GPUMutex;
use pumicite::tracking::{Access, ResourceState};
use pumicite::utils::AsVkHandle;
use pumicite::{Allocator, HasDevice, Sampler};

use crate::{HdrRenderTarget, HdrRenderTargetViewsExt};

/// Deepest chain the build shader can emit (`kMaxMips` in hzb_build.slang):
/// mip 0 up to 2048 texels, i.e. render resolution up to 4096.
pub const MAX_MIPS: u32 = 12;
/// Mip-0 texels covered by one workgroup of the build shader, per axis.
const TILE_SIZE: u32 = 64;

/// CPU mirror of `HzbConstants` in hzb_build.slang.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HzbConstantsRaw {
    render_extent: [u32; 2],
    hzb_extent: [u32; 2],
    num_mips: u32,
    num_groups: u32,
    inv_render_extent: [f32; 2],
}

pub struct HzbResources {
    /// R32_SFLOAT mip chain. `full_view()` samples every level (the trace);
    /// `mip_view(n)` is the storage view the build writes level `n` through.
    pub image: GPUMutex<MipImageViews<FullImageView<Image>>>,
    pub state: ResourceState,
    /// Last-workgroup election counter of the build shader. Zeroed once at
    /// creation; the elected group resets it every frame.
    pub counter: GPUMutex<Buffer>,
    pub counter_state: ResourceState,
    counter_needs_clear: bool,
    /// Mip-0 extent: half the power-of-two-rounded render extent, per axis.
    pub extent: UVec2,
    pub num_mips: u32,
    /// Render extent the chain was sized for.
    pub render_extent: UVec2,
}

impl HzbResources {
    /// `renderExtent / (2 * hzbExtent)`: scales a screen UV to an HZB UV.
    pub fn uv_factor(&self) -> Vec2 {
        self.render_extent.as_vec2() / (2.0 * self.extent.as_vec2())
    }
}

#[derive(Resource)]
pub struct HzbState {
    pub pipeline: Handle<ComputePipeline>,
    /// Gathers the depth G-buffer. Clamp-to-border with a transparent-black
    /// border, so depth texels past the render extent read as 0 (far).
    depth_sampler: Sampler,
    pub resources: Option<HzbResources>,
}

pub(crate) fn setup_hzb(mut commands: Commands, asset_server: Res<AssetServer>, allocator: Res<Allocator>) {
    let depth_sampler = Sampler::new(
        allocator.device().clone(),
        &vk::SamplerCreateInfo {
            mag_filter: vk::Filter::NEAREST,
            min_filter: vk::Filter::NEAREST,
            mipmap_mode: vk::SamplerMipmapMode::NEAREST,
            address_mode_u: vk::SamplerAddressMode::CLAMP_TO_BORDER,
            address_mode_v: vk::SamplerAddressMode::CLAMP_TO_BORDER,
            address_mode_w: vk::SamplerAddressMode::CLAMP_TO_BORDER,
            border_color: vk::BorderColor::FLOAT_TRANSPARENT_BLACK,
            ..Default::default()
        },
    )
    .unwrap();
    commands.insert_resource(HzbState {
        pipeline: asset_server.load("bazel://dust/crates/pbr/shaders/hzb_build.comp.pipeline.bin"),
        depth_sampler,
        resources: None,
    });
}

fn create_hzb_resources(allocator: &Allocator, render_extent: UVec2) -> HzbResources {
    let extent = UVec2::new(
        (render_extent.x.next_power_of_two() / 2).max(1),
        (render_extent.y.next_power_of_two() / 2).max(1),
    );
    let num_mips = 32 - extent.max_element().leading_zeros();
    assert!(
        num_mips <= MAX_MIPS,
        "HZB of {extent} needs {num_mips} mips, the build shader emits at most {MAX_MIPS}"
    );
    let image = Image::new_device(
        allocator.clone(),
        &vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format: vk::Format::R32_SFLOAT,
            extent: vk::Extent3D {
                width: extent.x,
                height: extent.y,
                depth: 1,
            },
            mip_levels: num_mips,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_SRC,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        },
    )
    .unwrap()
    .with_name(c"HZB")
    .create_full_view()
    .unwrap()
    .with_name(c"HZB View")
    .create_mip_views()
    .unwrap();
    let counter = Buffer::new_device(
        allocator.clone(),
        4,
        4,
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
    )
    .unwrap();
    HzbResources {
        image: GPUMutex::new(image),
        state: ResourceState::default(),
        counter: GPUMutex::new(counter),
        counter_state: ResourceState::default(),
        counter_needs_clear: true,
        extent,
        num_mips,
        render_extent,
    }
}

pub(crate) fn ensure_hzb_resources(
    mut hzb: ResMut<HzbState>,
    hdr_target: Option<Res<HdrRenderTarget>>,
    allocator: Res<Allocator>,
) {
    let Some(hdr) = hdr_target else {
        return;
    };
    if hdr.render_extent.x == 0 || hdr.render_extent.y == 0 {
        return;
    }
    let stale = hzb
        .resources
        .as_ref()
        .is_none_or(|res| res.render_extent != hdr.render_extent);
    if stale {
        hzb.resources = Some(create_hzb_resources(&allocator, hdr.render_extent));
    }
}

pub(crate) fn hzb_build_pass(
    mut ctx: SubmissionState,
    mut hzb: ResMut<HzbState>,
    compute_pipelines: Res<Assets<ComputePipeline>>,
    mut hdr_target: Option<ResMut<HdrRenderTarget>>,
    mut uniform_ring_buffer: ResMut<UniformRingBuffer>,
    mut profiler: Option<ResMut<GpuProfiler>>,
) {
    let Some(hdr) = hdr_target.as_mut() else {
        return;
    };
    let Some(pipeline) = compute_pipelines.get(&hzb.pipeline) else {
        return;
    };
    let pipeline = pipeline.clone().into_inner();
    let depth_sampler = hzb.depth_sampler.vk_handle();
    let Some(res) = hzb.resources.as_mut() else {
        return;
    };
    let groups = UVec2::new(
        res.extent.x.div_ceil(TILE_SIZE),
        res.extent.y.div_ceil(TILE_SIZE),
    );
    let constants = HzbConstantsRaw {
        render_extent: hdr.render_extent.to_array(),
        hzb_extent: res.extent.to_array(),
        num_mips: res.num_mips,
        num_groups: groups.x * groups.y,
        inv_render_extent: (1.0 / hdr.render_extent.as_vec2()).to_array(),
    };
    let num_mips = res.num_mips;
    let clear_counter = std::mem::take(&mut res.counter_needs_clear);

    ctx.record(|encoder| {
        let constants_buffer =
            uniform_ring_buffer.create_uniform(encoder, bytemuck::bytes_of(&constants));
        let views = encoder.lock(&hdr.view, vk::PipelineStageFlags2::COMPUTE_SHADER);
        let image = encoder.lock(&res.image, vk::PipelineStageFlags2::COMPUTE_SHADER);
        let counter = encoder.lock(&res.counter, vk::PipelineStageFlags2::COMPUTE_SHADER);

        if clear_counter {
            encoder.use_buffer_resource(counter, &mut res.counter_state, Access::CLEAR);
            encoder.emit_barriers();
            let device = encoder.device().clone();
            unsafe {
                device.cmd_fill_buffer(encoder.buffer().vk_handle(), counter.vk_handle(), 0, 4, 0);
            }
        }

        encoder.use_image_resource(
            views.depth(),
            &mut hdr.depth_state,
            Access {
                stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                access: vk::AccessFlags2::SHADER_SAMPLED_READ,
            },
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        // Every level is rewritten, so last frame's chain can be discarded.
        encoder.use_image_resource(
            image,
            &mut res.state,
            Access {
                stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                access: vk::AccessFlags2::SHADER_STORAGE_READ
                    | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            },
            vk::ImageLayout::GENERAL,
            0..num_mips,
            0..1,
            true,
        );
        encoder.use_buffer_resource(
            counter,
            &mut res.counter_state,
            Access {
                stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                access: vk::AccessFlags2::SHADER_STORAGE_READ
                    | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            },
        );
        encoder.emit_barriers();

        let pipeline = encoder.retain(pipeline).deref_inner();
        encoder.bind_pipeline(vk::PipelineBindPoint::COMPUTE, pipeline);

        // Bindings mirror the `[vk::binding]` slots in hzb_build.slang.
        let depth_info = [vk::DescriptorImageInfo {
            sampler: vk::Sampler::null(),
            image_view: views.depth().full_view().vk_handle(),
            image_layout: vk::ImageLayout::GENERAL,
        }];
        // Slots past the chain's end point at the last level: the shader never
        // touches them, and a real view avoids relying on null descriptors.
        let mip_infos: [vk::DescriptorImageInfo; MAX_MIPS as usize] =
            std::array::from_fn(|level| vk::DescriptorImageInfo {
                sampler: vk::Sampler::null(),
                image_view: image.mip_view((level as u32).min(num_mips - 1)).vk_handle(),
                image_layout: vk::ImageLayout::GENERAL,
            });
        let mip6_info = [mip_infos[6.min(num_mips as usize - 1)]];
        let counter_info = [vk::DescriptorBufferInfo {
            buffer: counter.vk_handle(),
            offset: 0,
            range: counter.size(),
        }];
        let sampler_info = [vk::DescriptorImageInfo {
            sampler: depth_sampler,
            image_view: vk::ImageView::null(),
            image_layout: vk::ImageLayout::UNDEFINED,
        }];
        let constants_info = [vk::DescriptorBufferInfo {
            buffer: constants_buffer.vk_handle(),
            offset: constants_buffer.offset(),
            range: constants_buffer.size(),
        }];
        encoder.push_descriptor_set(
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout(),
            0,
            &[
                vk::WriteDescriptorSet {
                    dst_binding: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    ..Default::default()
                }
                .image_info(&depth_info),
                vk::WriteDescriptorSet {
                    dst_binding: 1,
                    descriptor_count: MAX_MIPS,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    ..Default::default()
                }
                .image_info(&mip_infos),
                vk::WriteDescriptorSet {
                    dst_binding: 2,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    ..Default::default()
                }
                .image_info(&mip6_info),
                vk::WriteDescriptorSet {
                    dst_binding: 3,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    ..Default::default()
                }
                .buffer_info(&counter_info),
                vk::WriteDescriptorSet {
                    dst_binding: 4,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    ..Default::default()
                }
                .buffer_info(&constants_info),
                vk::WriteDescriptorSet {
                    dst_binding: 5,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::SAMPLER,
                    ..Default::default()
                }
                .image_info(&sampler_info),
            ],
        );
        encoder.timing_scope(profiler.as_deref_mut(), "hzb build", |encoder| {
            encoder.dispatch(UVec3::new(groups.x, groups.y, 1));
        });
    });
}
