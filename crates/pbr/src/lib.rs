use std::{
    alloc::Layout,
    collections::{BTreeMap, BTreeSet},
    ops::Deref,
    sync::Arc,
};

use bevy::prelude::*;
use rhyolite::{HasDevice, ash::vk::{self, TaggedStructure}, bevy::PipelineCache, image::ImageLike, shader::ShaderBindingTable, tracking::Access, utils::AsVkHandle};
use rhyolite_bevy::{
    DefaultRenderSet, RenderSetSharedStateWrapper,
    rtx::{RayTracingPipeline, RtxPipelineManager, tlas::TLAS},
    shader::RayTracingPipelineLibrary,
    staging::Uploader, swapchain::SwapchainImage,
};

pub struct PbrRenderPlugin;
impl Plugin for PbrRenderPlugin {
    fn build(&self, app: &mut App) {
        use bevy::asset::embedded_asset;
        embedded_asset!(app, "shaders/pbr.spv");
        embedded_asset!(app, "shaders/pbr.rtx.pipeline.ron");

        app.add_systems(Startup, setup);
        app.add_systems(PostUpdate, (
            create_sbt.before(PbrRenderSet),
            render.in_set(DefaultRenderSet)
            .after(PbrRenderSet)
            .after(rhyolite_bevy::rtx::tlas::tlas_build_system::<()>)
            .after(rhyolite_bevy::rtx::build_rtx_pipeline_system)
            .after(create_sbt)
        ));
        app.add_plugins(rhyolite_bevy::rtx::RtxPipelinePlugin);
        
        // Build a TLAS over everything.
        app.add_plugins(rhyolite_bevy::rtx::tlas::TLASBuilderPlugin::<()>::default());
    }
}

/// All the systems preparing for PBR raytracing render must go into this set.
#[derive(SystemSet, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Clone)]
pub struct PbrRenderSet;

fn render(
    mut ctx: RenderSetSharedStateWrapper,
    mut state: ResMut<PbrRenderState>,
    tlas: Res<TLAS>,
    pipelines: Res<Assets<RayTracingPipeline>>,
    mut uploader: Uploader,
    mut swapchain_images: Query<&mut SwapchainImage, With<bevy::window::PrimaryWindow>>,
) {
    let Ok(mut swapchain_image) = swapchain_images.single_mut() else {
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
    ctx.record(move |encoder| {
        let sbt_buffer = uploader.create_preinitialized_buffer_retained(
            encoder,
            sbt.layout(),
            |slice| slice.copy_from_slice(sbt.buffer()),
        );
        let pipeline = encoder.retain(pipeline);
        let tlas = encoder.lock(tlas, vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR);
        encoder.bind_pipeline(vk::PipelineBindPoint::RAY_TRACING_KHR, &pipeline);

        let tlas = encoder.retain(Box::new(tlas));
        let target_image = encoder.lock(swapchain_image.inner.as_ref().unwrap(), vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR);

        encoder.use_image_resource(
            target_image,
            &mut swapchain_image.state,
            Access::RTX_WRITE,
            vk::ImageLayout::GENERAL, 0..1, 0..1, true);

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
                ])
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
