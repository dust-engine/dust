use std::{
    alloc::Layout,
    collections::{BTreeMap, BTreeSet},
    ops::Deref,
    sync::Arc,
};

use bevy::prelude::*;
use rhyolite::{HasDevice, ash::vk, bevy::PipelineCache, shader::ShaderBindingTable};
use rhyolite_bevy::{
    DefaultRenderSet, RenderSetSharedStateWrapper,
    rtx::{RayTracingPipeline, RtxPipelineManager, tlas::TLAS},
    shader::RayTracingPipelineLibrary,
    staging::Uploader,
};

pub struct PbrRenderPlugin;
impl Plugin for PbrRenderPlugin {
    fn build(&self, app: &mut App) {
        use bevy::asset::embedded_asset;
        embedded_asset!(app, "shaders/pbr.spv");
        embedded_asset!(app, "shaders/pbr.rtx.pipeline.ron");

        app.add_systems(Startup, setup);
        app.add_systems(PostUpdate, (create_sbt, render.in_set(DefaultRenderSet)));
        app.add_plugins(rhyolite_bevy::rtx::RtxPipelinePlugin);
    }
}

fn render(
    mut ctx: RenderSetSharedStateWrapper,
    state: Res<PbrRenderState>,
    tlas: Res<TLAS>,
    pipelines: Res<Assets<RayTracingPipeline>>,
    mut uploader: Uploader,
) {
    let Some(pipeline) = pipelines.get(&state.pipeline).map(|x| x.deref().clone()) else {
        return;
    };
    let Some(sbt) = state.sbt.as_ref() else {
        return;
    };
    let Some(tlas) = tlas.get() else {
        return;
    };
    ctx.record(move |encoder| {
        let sbt_buffer = uploader.create_preinitialized_buffer_retained(
            encoder,
            sbt.layout(),
            |slice| slice.copy_from_slice(sbt.buffer()),
        );
        let pipeline = encoder.retain(pipeline);
        let tlas = encoder.lock(tlas, vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR);
        encoder.bind_pipeline(vk::PipelineBindPoint::RAY_TRACING_KHR, &pipeline);
        encoder.trace_rays(sbt, sbt_buffer, UVec3::ZERO);
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
