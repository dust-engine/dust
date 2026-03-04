pub mod camera;
pub mod sky;

use std::{
    alloc::Layout,
    collections::{BTreeMap, BTreeSet},
    ops::Deref,
    sync::Arc,
};

use bevy::prelude::*;
use bytemuck::{Pod, Zeroable};
use pumicite::{HasDevice, ash::vk::{self, TaggedStructure}, bevy::PipelineCache, buffer::BufferLike, image::ImageLike, rtx::ShaderBindingTable, tracking::Access, utils::{AsVkHandle, glam_to_vk_transform}};
use bevy_pumicite::{
    DefaultRenderSet, PumiciteApp, SubmissionState, rtx::{RayTracingPipeline, RtxPipelineManager, tlas::TLAS}, shader::RayTracingPipelineLibrary, staging::{BufferInitializer, UniformRingBuffer}, swapchain::SwapchainImage
};
use pumicite_egui::EguiRenderSet;

use crate::camera::Camera;

#[repr(C)]
#[derive(Clone, Copy)]
struct Uniforms {
    /// row-major affine transformation matrix.
    model: vk::TransformMatrixKHR,
    tan_half_fov: f32,
    far: f32,
    near: f32,
}
unsafe impl Pod for Uniforms{}
unsafe impl Zeroable for Uniforms{}

pub struct PbrRenderPlugin;
impl Plugin for PbrRenderPlugin {
    fn build(&self, app: &mut App) {
        use bevy::asset::embedded_asset;
        embedded_asset!(app, "shaders/pbr.spv");
        embedded_asset!(app, "shaders/pbr.rtx.pipeline.ron");
        embedded_asset!(app, "shaders/pbr.playout.ron");

        app.add_systems(Startup, setup);
        app.add_systems(PostUpdate, (
            create_sbt.before(PbrRenderSet),
            render.in_set(DefaultRenderSet)
            .after(PbrRenderSet)
            .after(bevy_pumicite::rtx::tlas::tlas_build_system::<()>)
            .after(bevy_pumicite::rtx::build_rtx_pipeline_system)
            .after(OccludingRenderPass)
            .after(create_sbt)
        ));
        app.add_plugins(bevy_pumicite::rtx::RtxPipelinePlugin);

        app.add_render_set(OccludingRenderPass, start_occluding_render_pass);
        app.configure_sets(PostUpdate, OccludingRenderPass.in_set(DefaultRenderSet));
        app.configure_sets(PostUpdate, EguiRenderSet.in_set(OccludingRenderPass));

        
        // Build a TLAS over everything.
        app.add_plugins(bevy_pumicite::rtx::tlas::TLASBuilderPlugin::<()>::default());

        app.add_plugins(sky::SkyPlugin);
    }
}

/// All the systems preparing for PBR raytracing render must go into this set.
#[derive(SystemSet, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Clone)]
pub struct PbrRenderSet;

fn render(
    mut ctx: SubmissionState,
    mut state: ResMut<PbrRenderState>,
    tlas: Res<TLAS>,
    pipelines: Res<Assets<RayTracingPipeline>>,
    mut uploader: BufferInitializer,
    mut uniform_ring_buffer: ResMut<UniformRingBuffer>,
    mut swapchain_images: Query<(&mut SwapchainImage, &Camera, &GlobalTransform), With<bevy::window::PrimaryWindow>>,
) {
    let Ok((mut swapchain_image, camera, transform)) = swapchain_images.single_mut() else {
        tracing::warn!("Frame not rendered; missing swapchain image");
        return;
    };
    let Some(pipeline) = pipelines.get(&state.pipeline).map(|x| x.deref().clone()) else {
        tracing::warn!("Frame not rendered; missing pipeline");
        return;
    };
    let Some(sbt) = state.sbt.as_mut() else {
        tracing::warn!("Frame not rendered; missing SBT");
        return;
    };
    let Some(tlas) = tlas.get() else {
        tracing::warn!("Frame not rendered; missing TLAS");
        return;
    };
    sbt.push_raygen(0, |_|{});
    sbt.push_miss(0, |_| {});
    let uniform = Uniforms {
        far: camera.depth.end,
        near: camera.depth.start,
        model: glam_to_vk_transform(transform.affine()),
        tan_half_fov: camera.tan_half_fov()
    };
    ctx.record(move |encoder| {
        let sbt_buffer = uploader.create_preinitialized_buffer_retained(
            encoder,
            sbt.layout(),
            |slice| slice.copy_from_slice(sbt.buffer()),
        );


        let uniform = uniform_ring_buffer.create_uniform(encoder, bytemuck::bytes_of(&uniform));

        let pipeline = encoder.retain(pipeline);
        let tlas = encoder.lock(tlas, vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR);
        encoder.bind_pipeline(vk::PipelineBindPoint::RAY_TRACING_KHR, &pipeline);

        let tlas = encoder.retain(Box::new(tlas));
        let target_image = encoder.lock(swapchain_image.current_image().as_ref().unwrap(), vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR);

        encoder.use_image_resource(
            target_image,
            &mut swapchain_image.state,
            Access::RTX_WRITE,
            vk::ImageLayout::GENERAL, 0..1, 0..1, true);
        encoder.memory_barrier(Access::COPY_WRITE, Access {
            stage: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
            access: vk::AccessFlags2::SHADER_BINDING_TABLE_READ_KHR,
        });

        encoder.emit_barriers();


        encoder.push_descriptor_set(
            vk::PipelineBindPoint::RAY_TRACING_KHR,
            pipeline.layout(),
            0,
            &[
                vk::WriteDescriptorSet {
                    dst_binding: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                    ..Default::default()
                }.push(&mut vk::WriteDescriptorSetAccelerationStructureKHR::default().acceleration_structures(&[tlas.vk_handle()])),
                vk::WriteDescriptorSet {
                    dst_binding: 1,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    ..Default::default()
                }.image_info(&[
                    vk::DescriptorImageInfo {
                        image_layout: vk::ImageLayout::GENERAL,
                        image_view: target_image.linear_view().vk_handle(),
                        ..Default::default()
                    }
                ]),
                vk::WriteDescriptorSet {
                    dst_binding: 2,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    ..Default::default()
                }.buffer_info(&[
                    vk::DescriptorBufferInfo {
                        buffer: uniform.vk_handle(),
                        offset: uniform.offset(),
                        range: uniform.size(),
                    }
                ]),
            ]
        );
        encoder.trace_rays(sbt, 0, sbt_buffer, target_image.extent());
    });
}

#[derive(Resource)]
pub struct PbrRenderState {
    pub pipeline: Handle<RayTracingPipeline>,
    pub sbt: Option<ShaderBindingTable>,
}

pub fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut pipeline_manager: ResMut<RtxPipelineManager>,
) {
    let base_library: Handle<RayTracingPipelineLibrary> =
        asset_server.load("embedded://dust_pbr/shaders/pbr.rtx.pipeline.ron");

    commands.insert_resource(PbrRenderState {
        pipeline: pipeline_manager.add_pipeline(base_library),
        sbt: None,
    });
}

pub fn create_sbt(mut state: ResMut<PbrRenderState>, pipelines: Res<Assets<RayTracingPipeline>>) {
    let Some(pipeline) = pipelines.get(&state.pipeline) else {
        state.sbt = None;
        return;
    };
    state.sbt = Some(pipeline.create_sbt(state.sbt.take()));
}

#[derive(Default, SystemSet, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Clone, Copy)]
pub struct OccludingRenderPass;


/// For egui, you are always responsible for setting up the render pass. This
/// is so that egui can piggy-back on an already existing render pass and doesn't
/// have to start a new one - important for mobile performance.
fn start_occluding_render_pass(
    mut ctx: SubmissionState,
    mut swapchain_image: Query<&mut SwapchainImage, With<bevy::window::PrimaryWindow>>,
) {
    let Ok(mut swapchain_image) = swapchain_image.single_mut() else {
        return;
    };
    ctx.record(|encoder| {
        let Some(current_swapchain_image) = swapchain_image.current_image() else {
            return;
        };
        let current_swapchain_image = encoder.lock(
            current_swapchain_image,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        );

        encoder.use_image_resource(
            current_swapchain_image,
            &mut swapchain_image.state,
            Access::COLOR_ATTACHMENT_WRITE,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            0..1,
            0..1,
            false,
        );
        encoder.emit_barriers();
        encoder
            .begin_rendering()
            .color_attachment(0, |mut builder| {
                builder
                    .clear(Vec4::new(0.0, 0.0, 0.0, 0.0))
                    .image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
                    .store(true)
                    .view(current_swapchain_image.linear_view());
                // Use linear view for egui. egui does all the interpolation in srgb space.
            })
            .render_area(IVec2::ZERO, current_swapchain_image.extent().xy())
            .begin();
    });
}
