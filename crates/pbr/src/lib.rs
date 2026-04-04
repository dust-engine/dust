pub mod camera;
pub mod sky;

include!(concat!(
    env!("BAZEL_BIN"),
    "/crates/pbr/shaders/pbr_module_layout.rs"
));

use std::ops::Deref;

use bevy::prelude::*;
use bevy_pumicite::{
    DefaultRenderSet, PumiciteApp, SubmissionState,
    rtx::{RayTracingPipeline, RtxPipelineManager, tlas::TLAS},
    shader::RayTracingPipelineLibrary,
    staging::{BufferInitializer, UniformRingBuffer},
    swapchain::{SwapchainImage, SwapchainSet},
};
use bytemuck::{Pod, Zeroable};
use pumicite::{
    Allocator, HasDevice,
    ash::vk::{self, TaggedStructure},
    bevy::PipelineCache,
    buffer::BufferLike,
    debug::DebugObject,
    image::{FullImageView, Image, ImageExt, ImageLike, SrgbImageView},
    rtx::ShaderBindingTable,
    sync::GPUMutex,
    tracking::{Access, ResourceState},
    utils::{AsVkHandle, glam_to_vk_transform},
};
use pumicite_egui::EguiRenderSet;

use bevy_pumicite::prelude::ComputePipeline;

use crate::{
    camera::Camera,
    sky::{AtmosphereLUTs, AtmosphereState, SkyAtmosphereLUTRenderSet},
};

#[repr(C)]
#[derive(Clone, Copy)]
struct Uniforms {
    /// row-major affine transformation matrix.
    model: vk::TransformMatrixKHR,
    tan_half_fov: f32,
    far: f32,
    near: f32,
}
unsafe impl Pod for Uniforms {}
unsafe impl Zeroable for Uniforms {}

pub struct PbrRenderPlugin;
impl Plugin for PbrRenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup);
        app.add_systems(
            PostUpdate,
            (
                create_sbt.before(PbrRenderSet),
                ensure_hdr_target.in_set(SwapchainSet).before(render),
                render
                    .in_set(DefaultRenderSet)
                    .after(PbrRenderSet)
                    .after(bevy_pumicite::rtx::tlas::tlas_build_system::<()>)
                    .after(bevy_pumicite::rtx::build_rtx_pipeline_system)
                    .after(OccludingRenderPass)
                    .after(create_sbt),
                shadow_pass
                    .in_set(DefaultRenderSet)
                    .after(PbrRenderSet)
                    .after(render)
                    .after(create_sbt),
                tonemap_pass
                    .in_set(DefaultRenderSet)
                    .after(render)
                    .after(shadow_pass),
            ),
        );
        app.add_plugins(bevy_pumicite::rtx::RtxPipelinePlugin);

        app.add_render_set(OccludingRenderPass, start_occluding_render_pass);
        app.configure_sets(PostUpdate, OccludingRenderPass.in_set(DefaultRenderSet));
        app.configure_sets(PostUpdate, EguiRenderSet.in_set(OccludingRenderPass));

        // Build a TLAS over everything.
        app.add_plugins(bevy_pumicite::rtx::tlas::TLASBuilderPlugin::<()>::default());

        app.add_plugins(sky::SkyPlugin);
        app.configure_sets(
            PostUpdate,
            SkyAtmosphereLUTRenderSet
                .in_set(DefaultRenderSet)
                .before(render),
        );
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
    mut swapchain_images: Query<
        (&mut SwapchainImage, &Camera, &GlobalTransform),
        With<bevy::window::PrimaryWindow>,
    >,
    atmosphere: Res<AtmosphereState>,
    mut luts: ResMut<AtmosphereLUTs>,
    mut hdr_target: Option<ResMut<HdrRenderTarget>>,
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
    let Some(hdr) = hdr_target.as_mut() else {
        tracing::warn!("Frame not rendered; missing HDR target");
        return;
    };
    sbt.push_raygen(0, |_| {});
    sbt.push_miss(0, |_| {});
    let uniform = Uniforms {
        far: camera.depth.end,
        near: camera.depth.start,
        model: glam_to_vk_transform(transform.affine()),
        tan_half_fov: camera.tan_half_fov(),
    };
    let atmosphere_uniform_buffer = atmosphere.uniform_buffer.as_ref().unwrap().clone();
    ctx.record(move |encoder| {
        let sbt_buffer =
            uploader.create_preinitialized_buffer_retained(encoder, sbt.layout(), |slice| {
                slice.copy_from_slice(sbt.buffer())
            });

        let uniform = uniform_ring_buffer.create_uniform(encoder, bytemuck::bytes_of(&uniform));
        let atmo_buffer = encoder.retain(atmosphere_uniform_buffer);

        let pipeline = encoder.retain(pipeline);
        let tlas = encoder.lock(tlas, vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR);
        encoder.bind_pipeline(vk::PipelineBindPoint::RAY_TRACING_KHR, &pipeline);

        // HDR intermediary at binding 1
        let hdr_image = encoder.lock(&hdr.view, vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR);

        // G-buffer textures at bindings 8, 9, 10
        let albedo_image = encoder.lock(
            &hdr.albedo_view,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );
        let normal_image = encoder.lock(
            &hdr.normal_view,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );
        let depth_image = encoder.lock(
            &hdr.depth_view,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );

        // Swapchain at binding 7 (read-only for occlusion alpha check)
        let swapchain_target = encoder.lock(
            swapchain_image.current_image().as_ref().unwrap(),
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );

        // Lock sky atmosphere LUTs for reading in the miss shader
        let transmittance_view = encoder.lock(
            &luts.transmittance.view,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );
        let sky_view = encoder.lock(
            &luts.sky_view.view,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );

        // HDR target: write (discard previous contents)
        encoder.use_image_resource(
            hdr_image.image(),
            &mut hdr.state,
            Access::RTX_WRITE,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            true,
        );
        // Albedo G-buffer: write (discard previous contents)
        encoder.use_image_resource(
            albedo_image.image(),
            &mut hdr.albedo_state,
            Access::RTX_WRITE,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            true,
        );
        // Normal G-buffer: write (discard previous contents)
        encoder.use_image_resource(
            normal_image.image(),
            &mut hdr.normal_state,
            Access::RTX_WRITE,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            true,
        );
        // Depth G-buffer: write (discard previous contents)
        encoder.use_image_resource(
            depth_image.image(),
            &mut hdr.depth_state,
            Access::RTX_WRITE,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            true,
        );
        // Swapchain: read-only for occlusion check (preserve egui content)
        encoder.use_image_resource(
            swapchain_target,
            &mut swapchain_image.state,
            Access {
                stage: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                access: vk::AccessFlags2::SHADER_STORAGE_READ,
            },
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            transmittance_view.image(),
            &mut luts.transmittance.state,
            Access::RTX_READ,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            sky_view.image(),
            &mut luts.sky_view.state,
            Access::RTX_READ,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            0..1,
            0..1,
            false,
        );
        encoder.memory_barrier(
            Access::COPY_WRITE,
            Access {
                stage: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                access: vk::AccessFlags2::SHADER_BINDING_TABLE_READ_KHR,
            },
        );

        encoder.emit_barriers();

        encoder.push_descriptor_set(
            vk::PipelineBindPoint::RAY_TRACING_KHR,
            pipeline.layout(),
            0,
            PbrPipelineParams::new()
                .scene_bvh(tlas)
                .uniforms(uniform)
                .output_texture(hdr_image)
                .gbuffer_albedo_linear(&albedo_image.linear_view())
                .gbuffer_albedo_srgb(&albedo_image.srgb_view())
                .gbuffer_normal_texture(normal_image)
                .gbuffer_depth_texture(depth_image)
                .sky_atmosphere_params(atmo_buffer)
                .sky_transmittance_lut(transmittance_view)
                .sky_sky_view_lut(sky_view)
                .sky_linear_sampler(&luts.sampler)
                .as_slice(),
        );
        encoder.trace_rays(sbt, 0, sbt_buffer, hdr_image.image().extent());
    });
}

#[derive(Resource)]
pub struct HdrRenderTarget {
    pub view: GPUMutex<FullImageView<Image>>,
    pub state: ResourceState,
    pub albedo_view: GPUMutex<SrgbImageView<Image>>,
    pub albedo_state: ResourceState,
    pub normal_view: GPUMutex<FullImageView<Image>>,
    pub normal_state: ResourceState,
    pub depth_view: GPUMutex<FullImageView<Image>>,
    pub depth_state: ResourceState,
    pub extent: UVec2,
}

#[derive(Resource)]
pub struct PbrRenderState {
    pub pipeline: Handle<RayTracingPipeline>,
    pub shadow_pipeline: Handle<RayTracingPipeline>,
    pub tonemap_pipeline: Handle<ComputePipeline>,
    pub sbt: Option<ShaderBindingTable>,
    pub shadow_sbt: Option<ShaderBindingTable>,
}

pub fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut pipeline_manager: ResMut<RtxPipelineManager>,
) {
    let base_library: Handle<RayTracingPipelineLibrary> =
        asset_server.load("bazel://crates/pbr/shaders:pbr.rtx.pipeline.bin");
    let shadow_base_library: Handle<RayTracingPipelineLibrary> =
        asset_server.load("bazel://crates/pbr/shaders:shadow.rtx.pipeline.bin");

    let tonemap_pipeline: Handle<ComputePipeline> =
        asset_server.load("bazel://crates/pbr/shaders:tonemap.comp.pipeline.bin");

    commands.insert_resource(PbrRenderState {
        pipeline: pipeline_manager.add_pipeline(base_library),
        shadow_pipeline: pipeline_manager.add_pipeline(shadow_base_library),
        tonemap_pipeline,
        sbt: None,
        shadow_sbt: None,
    });
}

pub fn create_sbt(mut state: ResMut<PbrRenderState>, pipelines: Res<Assets<RayTracingPipeline>>) {
    if let Some(pipeline) = pipelines.get(&state.pipeline) {
        state.sbt = Some(pipeline.create_sbt(state.sbt.take()));
    } else {
        state.sbt = None;
    }
    if let Some(pipeline) = pipelines.get(&state.shadow_pipeline) {
        state.shadow_sbt = Some(pipeline.create_sbt(state.shadow_sbt.take()));
    } else {
        state.shadow_sbt = None;
    }
}

fn ensure_hdr_target(
    mut commands: Commands,
    hdr_target: Option<Res<HdrRenderTarget>>,
    allocator: Res<Allocator>,
    swapchain_images: Query<&SwapchainImage, With<bevy::window::PrimaryWindow>>,
) {
    let Ok(swapchain_image) = swapchain_images.single() else {
        return;
    };
    let Some(current) = swapchain_image.current_image() else {
        return;
    };
    let extent = UVec2::new(current.extent().x, current.extent().y);

    if let Some(hdr) = hdr_target.as_ref() {
        if hdr.extent == extent {
            return;
        }
    }

    let create_info = vk::ImageCreateInfo {
        image_type: vk::ImageType::TYPE_2D,
        extent: vk::Extent3D {
            width: extent.x,
            height: extent.y,
            depth: 1,
        },
        mip_levels: 1,
        array_layers: 1,
        samples: vk::SampleCountFlags::TYPE_1,
        tiling: vk::ImageTiling::OPTIMAL,
        usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
        initial_layout: vk::ImageLayout::UNDEFINED,
        ..Default::default()
    };
    let view = GPUMutex::new(
        Image::new_private(
            allocator.clone(),
            &vk::ImageCreateInfo {
                format: vk::Format::R16G16B16A16_SFLOAT,
                ..create_info
            },
        )
        .unwrap()
        .with_name(c"HDR Render Target")
        .create_full_view()
        .unwrap()
        .with_name(c"HDR Render Target View"),
    );
    let albedo_view = GPUMutex::new(
        Image::new_private(
            allocator.clone(),
            &vk::ImageCreateInfo {
                flags: vk::ImageCreateFlags::MUTABLE_FORMAT,
                format: vk::Format::R8G8B8A8_UNORM,
                ..create_info
            }
            .push(
                &mut vk::ImageFormatListCreateInfo::default()
                    .view_formats(&[vk::Format::R8G8B8A8_SRGB, vk::Format::R8G8B8A8_UNORM]),
            ),
        )
        .unwrap()
        .with_name(c"G-Buffer Albedo Image")
        .create_srgb_view(vk::ImageUsageFlags::STORAGE, vk::ImageUsageFlags::SAMPLED)
        .unwrap(),
    );
    let normal_view = GPUMutex::new(
        Image::new_private(
            allocator.clone(),
            &vk::ImageCreateInfo {
                format: vk::Format::R16G16B16A16_SFLOAT,
                ..create_info
            },
        )
        .unwrap()
        .with_name(c"G-Buffer Normal Image")
        .create_full_view()
        .unwrap()
        .with_name(c"G-Buffer Normal Image View"),
    );
    let depth_view = GPUMutex::new(
        Image::new_private(
            allocator.clone(),
            &vk::ImageCreateInfo {
                format: vk::Format::R32_SFLOAT,
                ..create_info
            },
        )
        .unwrap()
        .with_name(c"G-Buffer Depth")
        .create_full_view()
        .unwrap()
        .with_name(c"G-Buffer Depth View"),
    );

    commands.insert_resource(HdrRenderTarget {
        view,
        state: Default::default(),
        albedo_view,
        albedo_state: Default::default(),
        normal_view,
        normal_state: Default::default(),
        depth_view,
        depth_state: Default::default(),
        extent,
    });
}

fn shadow_pass(
    mut ctx: SubmissionState,
    mut state: ResMut<PbrRenderState>,
    tlas: Res<TLAS>,
    pipelines: Res<Assets<RayTracingPipeline>>,
    mut uploader: BufferInitializer,
    mut uniform_ring_buffer: ResMut<UniformRingBuffer>,
    swapchain_images: Query<(&Camera, &GlobalTransform), With<bevy::window::PrimaryWindow>>,
    atmosphere: Res<AtmosphereState>,
    mut luts: ResMut<AtmosphereLUTs>,
    mut hdr_target: Option<ResMut<HdrRenderTarget>>,
) {
    let Ok((camera, transform)) = swapchain_images.single() else {
        return;
    };
    let Some(pipeline) = pipelines
        .get(&state.shadow_pipeline)
        .map(|x| x.deref().clone())
    else {
        return;
    };
    let Some(shadow_sbt) = state.shadow_sbt.as_mut() else {
        return;
    };
    let Some(tlas) = tlas.get() else {
        return;
    };
    let Some(hdr) = hdr_target.as_mut() else {
        return;
    };
    shadow_sbt.push_raygen(0, |_| {});
    shadow_sbt.push_miss(0, |_| {});
    let uniform = Uniforms {
        far: camera.depth.end,
        near: camera.depth.start,
        model: glam_to_vk_transform(transform.affine()),
        tan_half_fov: camera.tan_half_fov(),
    };
    let atmosphere_uniform_buffer = atmosphere.uniform_buffer.as_ref().unwrap().clone();
    ctx.record(move |encoder| {
        let sbt_buffer =
            uploader.create_preinitialized_buffer_retained(encoder, shadow_sbt.layout(), |slice| {
                slice.copy_from_slice(shadow_sbt.buffer())
            });

        let uniform = uniform_ring_buffer.create_uniform(encoder, bytemuck::bytes_of(&uniform));
        let atmo_buffer = encoder.retain(atmosphere_uniform_buffer);

        let pipeline = encoder.retain(pipeline);
        let tlas = encoder.lock(tlas, vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR);
        encoder.bind_pipeline(vk::PipelineBindPoint::RAY_TRACING_KHR, &pipeline);

        // HDR output at binding 1 (read-write: miss shader adds sun contribution)
        let hdr_image = encoder.lock(&hdr.view, vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR);
        // Albedo at binding 8 (read)
        let albedo_image = encoder.lock(
            &hdr.albedo_view,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );
        // Normal G-buffer at binding 9 (read)
        let normal_image = encoder.lock(
            &hdr.normal_view,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );
        // Depth G-buffer at binding 10 (read)
        let depth_image = encoder.lock(
            &hdr.depth_view,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );

        // HDR: read-write (miss shader reads current value and adds sun contribution)
        encoder.use_image_resource(
            hdr_image.image(),
            &mut hdr.state,
            Access {
                stage: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                access: vk::AccessFlags2::SHADER_STORAGE_READ
                    | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            },
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        // Albedo: sampled read through sRGB view (written by primary pass)
        encoder.use_image_resource(
            albedo_image.image(),
            &mut hdr.albedo_state,
            Access {
                stage: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                access: vk::AccessFlags2::SHADER_SAMPLED_READ,
            },
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            0..1,
            0..1,
            false,
        );
        // Normal: read (written by primary pass)
        encoder.use_image_resource(
            normal_image.image(),
            &mut hdr.normal_state,
            Access {
                stage: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                access: vk::AccessFlags2::SHADER_STORAGE_READ,
            },
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        // Depth: read (written by primary pass)
        encoder.use_image_resource(
            depth_image.image(),
            &mut hdr.depth_state,
            Access {
                stage: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                access: vk::AccessFlags2::SHADER_STORAGE_READ,
            },
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        // Lock sky atmosphere LUTs for reading in the miss shader
        let transmittance_view = encoder.lock(
            &luts.transmittance.view,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );
        let sky_view = encoder.lock(
            &luts.sky_view.view,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );

        encoder.use_image_resource(
            transmittance_view.image(),
            &mut luts.transmittance.state,
            Access::RTX_READ,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            sky_view.image(),
            &mut luts.sky_view.state,
            Access::RTX_READ,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            0..1,
            0..1,
            false,
        );
        encoder.memory_barrier(
            Access::COPY_WRITE,
            Access {
                stage: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                access: vk::AccessFlags2::SHADER_BINDING_TABLE_READ_KHR,
            },
        );

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
                }
                .push(
                    &mut vk::WriteDescriptorSetAccelerationStructureKHR::default()
                        .acceleration_structures(&[tlas.vk_handle()]),
                ),
                vk::WriteDescriptorSet {
                    dst_binding: 1,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    ..Default::default()
                }
                .buffer_info(&[vk::DescriptorBufferInfo {
                    buffer: uniform.vk_handle(),
                    offset: uniform.offset(),
                    range: uniform.size(),
                }]),
                // Binding 2: HDR output texture (write)
                vk::WriteDescriptorSet {
                    dst_binding: 2,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    ..Default::default()
                }
                .image_info(&[vk::DescriptorImageInfo {
                    image_layout: vk::ImageLayout::GENERAL,
                    image_view: hdr_image.vk_handle(),
                    ..Default::default()
                }]),
                // Binding 3: Albedo G-buffer (write, linear/UNORM view)
                // Omitted
                // Binding 4: Albedo G-buffer (write, srgb view)
                vk::WriteDescriptorSet {
                    dst_binding: 4,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    ..Default::default()
                }
                .image_info(&[vk::DescriptorImageInfo {
                    image_layout: vk::ImageLayout::GENERAL,
                    image_view: albedo_image.srgb_view().vk_handle(),
                    ..Default::default()
                }]),
                // Binding 5: Normal G-buffer (write)
                vk::WriteDescriptorSet {
                    dst_binding: 5,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    ..Default::default()
                }
                .image_info(&[vk::DescriptorImageInfo {
                    image_layout: vk::ImageLayout::GENERAL,
                    image_view: normal_image.vk_handle(),
                    ..Default::default()
                }]),
                // Binding 6: Depth G-buffer (write)
                vk::WriteDescriptorSet {
                    dst_binding: 6,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    ..Default::default()
                }
                .image_info(&[vk::DescriptorImageInfo {
                    image_layout: vk::ImageLayout::GENERAL,
                    image_view: depth_image.vk_handle(),
                    ..Default::default()
                }]),
                vk::WriteDescriptorSet {
                    dst_binding: 8,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    ..Default::default()
                }
                .buffer_info(&[vk::DescriptorBufferInfo {
                    buffer: atmo_buffer.vk_handle(),
                    offset: atmo_buffer.offset(),
                    range: atmo_buffer.size(),
                }]),
                vk::WriteDescriptorSet {
                    dst_binding: 9,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    p_image_info: &vk::DescriptorImageInfo {
                        image_view: transmittance_view.vk_handle(),
                        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        sampler: vk::Sampler::null(),
                    },
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_binding: 10,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    p_image_info: &vk::DescriptorImageInfo {
                        image_view: sky_view.vk_handle(),
                        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        sampler: vk::Sampler::null(),
                    },
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_binding: 11,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::SAMPLER,
                    p_image_info: &vk::DescriptorImageInfo {
                        sampler: luts.sampler.vk_handle(),
                        ..Default::default()
                    },
                    ..Default::default()
                },
            ],
        );
        encoder.trace_rays(shadow_sbt, 0, sbt_buffer, normal_image.image().extent());
    });
}

fn tonemap_pass(
    mut ctx: SubmissionState,
    state: Res<PbrRenderState>,
    compute_pipelines: Res<Assets<ComputePipeline>>,
    mut swapchain_images: Query<(&mut SwapchainImage, &Camera), With<bevy::window::PrimaryWindow>>,
    mut hdr_target: Option<ResMut<HdrRenderTarget>>,
) {
    let Ok((mut swapchain_image, camera)) = swapchain_images.single_mut() else {
        return;
    };
    let Some(pipeline) = compute_pipelines.get(&state.tonemap_pipeline) else {
        return;
    };
    let Some(hdr) = hdr_target.as_mut() else {
        return;
    };
    let exposure = camera.exposure;
    let pipeline = pipeline.clone().into_inner();

    ctx.record(move |encoder| {
        let hdr_image = encoder.lock(&hdr.view, vk::PipelineStageFlags2::COMPUTE_SHADER);
        let swapchain_target = encoder.lock(
            swapchain_image.current_image().as_ref().unwrap(),
            vk::PipelineStageFlags2::COMPUTE_SHADER,
        );

        // HDR intermediary: read
        encoder.use_image_resource(
            hdr_image.image(),
            &mut hdr.state,
            Access::COMPUTE_READ,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        // Swapchain: write (preserve egui pixels — not discarded)
        encoder.use_image_resource(
            swapchain_target,
            &mut swapchain_image.state,
            Access::COMPUTE_WRITE,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        encoder.emit_barriers();

        let pipeline = encoder.retain(pipeline);
        encoder.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &pipeline);

        encoder.push_descriptor_set(
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout(),
            0,
            &[
                vk::WriteDescriptorSet {
                    dst_binding: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    ..Default::default()
                }
                .image_info(&[vk::DescriptorImageInfo {
                    image_layout: vk::ImageLayout::GENERAL,
                    image_view: hdr_image.vk_handle(),
                    ..Default::default()
                }]),
                vk::WriteDescriptorSet {
                    dst_binding: 1,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    ..Default::default()
                }
                .image_info(&[vk::DescriptorImageInfo {
                    image_layout: vk::ImageLayout::GENERAL,
                    image_view: swapchain_target.linear_view().vk_handle(),
                    ..Default::default()
                }]),
            ],
        );

        encoder.push_constants(
            pipeline.layout(),
            vk::ShaderStageFlags::COMPUTE,
            0,
            unsafe {
                std::slice::from_raw_parts(
                    &exposure as *const f32 as *const u8,
                    std::mem::size_of_val(&exposure),
                )
            },
        );

        let extent = hdr_image.image().extent();
        encoder.dispatch(UVec3::new(extent.x.div_ceil(8), extent.y.div_ceil(8), 1));
    });
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
