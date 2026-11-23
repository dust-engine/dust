//! The reference accumulator, the [`UpscalerBackend::Reference`] backend: the
//! running mean of every frame's radiance since it last restarted, at display
//! resolution (reference_accumulate.slang). It lets a noisy renderer converge
//! to its expected image, without a denoiser's bias. The camera jitters across
//! each pixel ([`JITTER_PHASES`]), so the mean is also antialiased with a box
//! filter.
//!
//! A debugging tool: it neither reprojects nor notices that the view or scene
//! changed, and keeps averaging. Write [`ResetHistory`] after a change, and
//! pause `Time<Virtual>` to hold a moving scene still.
//!
//! [`UpscalerBackend::Reference`]: super::UpscalerBackend::Reference
//! [`ResetHistory`]: super::ResetHistory

use std::sync::Arc;

use bevy::prelude::*;
use bevy_pumicite::prelude::ComputePipeline;
use bevy_pumicite::staging::UniformRingBuffer;
use bytemuck::{Pod, Zeroable};
use pumicite::Allocator;
use pumicite::ash::vk;
use pumicite::command::CommandEncoder;
use pumicite::debug::DebugObject;
use pumicite::image::{FullImageView, Image, ImageExt};
use pumicite::pipeline::Pipeline;
use pumicite::sync::GPUMutex;
use pumicite::tracking::{Access, ResourceState};
use pumicite::utils::AsVkHandle;

use super::{UpscalerInputs, UpscalerSession, UpscalerSessionState};

/// Sub-pixel jitter offsets the accumulator cycles through: enough Halton
/// (2, 3) points that a mean over thousands of frames covers each pixel's area
/// evenly.
pub(super) const JITTER_PHASES: u32 = 1024;

/// The running mean, the [`UpscalerSession`] of the reference backend.
pub struct ReferenceAccumulator {
    pipeline: Handle<ComputePipeline>,
    /// R32G32B32A32_SFLOAT, display resolution: the mean so far.
    image: GPUMutex<FullImageView<Image>>,
    state: ResourceState,
    /// Frames averaged into `image`. 0: its contents are undefined, and the
    /// next frame starts the mean.
    frames: u32,
}

impl ReferenceAccumulator {
    /// The accumulator, if it's the backend running.
    pub fn running(state: &UpscalerSessionState) -> Option<&Self> {
        match &state.session {
            Some(UpscalerSession::Reference(accumulator)) => Some(accumulator),
            _ => None,
        }
    }

    /// How many frames the mean has averaged.
    pub fn frames(&self) -> u32 {
        self.frames
    }

    pub(super) fn new(
        allocator: &Allocator,
        asset_server: &AssetServer,
        extent: UVec2,
    ) -> Result<Self, String> {
        let image = Image::new_device(
            allocator.clone(),
            &vk::ImageCreateInfo {
                image_type: vk::ImageType::TYPE_2D,
                format: vk::Format::R32G32B32A32_SFLOAT,
                extent: vk::Extent3D {
                    width: extent.x,
                    height: extent.y,
                    depth: 1,
                },
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::STORAGE,
                initial_layout: vk::ImageLayout::UNDEFINED,
                ..Default::default()
            },
        )
        .map_err(|e| format!("allocating the accumulation image failed: {e:?}"))?
        .with_name(c"Reference Accumulation")
        .create_full_view()
        .map_err(|e| format!("creating the accumulation image's view failed: {e:?}"))?
        .with_name(c"Reference Accumulation View");
        Ok(Self {
            pipeline: asset_server
                .load("bazel://dust/crates/pbr/shaders/reference_accumulate.comp.pipeline.bin"),
            image: GPUMutex::new(image),
            state: ResourceState::default(),
            frames: 0,
        })
    }

    /// The pipeline to [`dispatch`](Self::dispatch) with, once it has loaded.
    pub(super) fn pipeline(&self, pipelines: &Assets<ComputePipeline>) -> Option<Arc<Pipeline>> {
        pipelines
            .get(&self.pipeline)
            .map(|pipeline| pipeline.clone().into_inner())
    }

    /// Adds `inputs.color` to the mean, and writes the mean to `inputs.output`.
    pub(super) fn dispatch(
        &mut self,
        encoder: &mut CommandEncoder,
        inputs: &UpscalerInputs,
        pipeline: Arc<Pipeline>,
        uniform_ring_buffer: &mut UniformRingBuffer,
    ) {
        if inputs.reset_history {
            self.frames = 0;
        }
        let extent = inputs.output.extent;
        let constants = AccumulateConstants {
            sample_count: self.frames,
            _pad: 0,
            extent: extent.to_array(),
        };
        let constants = uniform_ring_buffer.create_uniform(encoder, bytemuck::bytes_of(&constants));
        let image = encoder.lock(&self.image, vk::PipelineStageFlags2::COMPUTE_SHADER);
        encoder.use_image_resource(
            image,
            &mut self.state,
            Access {
                stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                access: vk::AccessFlags2::SHADER_STORAGE_READ
                    | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            },
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            // A new mean doesn't read what the image held.
            self.frames == 0,
        );
        encoder.emit_barriers();

        let pipeline = encoder.retain(pipeline).deref_inner();
        encoder.bind_pipeline(vk::PipelineBindPoint::COMPUTE, pipeline);
        // Bindings mirror the `[vk::binding]` slots in
        // reference_accumulate.slang.
        let storage_image = |view: vk::ImageView| {
            [vk::DescriptorImageInfo {
                sampler: vk::Sampler::null(),
                image_view: view,
                image_layout: vk::ImageLayout::GENERAL,
            }]
        };
        let radiance_info = storage_image(inputs.color.view);
        let accumulation_info = storage_image(image.full_view().vk_handle());
        let output_info = storage_image(inputs.output.view);
        let constants_info = [vk::DescriptorBufferInfo {
            buffer: constants.vk_handle(),
            offset: constants.offset(),
            range: constants.size(),
        }];
        let storage_write = |binding: u32| vk::WriteDescriptorSet {
            dst_binding: binding,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
            ..Default::default()
        };
        encoder.push_descriptor_set(
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout(),
            0,
            &[
                storage_write(0).image_info(&radiance_info),
                storage_write(1).image_info(&accumulation_info),
                storage_write(2).image_info(&output_info),
                vk::WriteDescriptorSet {
                    dst_binding: 3,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    ..Default::default()
                }
                .buffer_info(&constants_info),
            ],
        );
        encoder.dispatch(UVec3::new(extent.x.div_ceil(8), extent.y.div_ceil(8), 1));
        self.frames = self.frames.saturating_add(1);
    }
}

/// CPU mirror of `AccumulateConstants` in reference_accumulate.slang.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AccumulateConstants {
    sample_count: u32,
    _pad: u32,
    extent: [u32; 2],
}
