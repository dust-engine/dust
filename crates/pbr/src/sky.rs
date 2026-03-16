//! Sky Atmosphere Example
//!
//! Implementation of "A Scalable and Production Ready Sky and Atmosphere Rendering Technique"
//! from EGSR 2020 by Sébastien Hillaire.
//!
//! Features:
//! - Precomputed Transmittance LUT (256x64)
//! - Multi-scattering LUT (32x32)
//! - Per-frame Sky View LUT (192x108)
//! - Interactive camera and sun controls
//! - egui parameter tweaking

use std::ffi::CStr;

use bevy::ecs::schedule::IntoScheduleConfigs;
use bevy::input::mouse::{MouseMotion, MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use bevy_pumicite::prelude::*;
use glam::{IVec2, Mat4, Vec3, Vec3Swizzles};
use pumicite::buffer::RingBufferSuballocation;
use pumicite::{
    Sampler,
    image::{FullImageView, Image},
};
use pumicite_egui::{EguiContexts, EguiPrimaryContextPass, EguiRenderSet, egui};

// LUT dimensions (must match shader constants)
const TRANSMITTANCE_WIDTH: u32 = 256;
const TRANSMITTANCE_HEIGHT: u32 = 64;
const MULTI_SCATTERING_RES: u32 = 32;
const SKY_VIEW_WIDTH: u32 = 192;
const SKY_VIEW_HEIGHT: u32 = 108;

pub struct SkyPlugin;

impl Plugin for SkyPlugin {
    fn build(&self, app: &mut App) {
        // Enable required extensions
        app.add_device_extension::<ash::khr::push_descriptor::Meta>()
            .unwrap();
        app.add_device_extension::<ash::khr::dynamic_rendering::Meta>()
            .unwrap();
        app.enable_feature::<vk::PhysicalDeviceDynamicRenderingFeatures>(|x| {
            &mut x.dynamic_rendering
        })
        .unwrap();
        app.enable_feature::<vk::PhysicalDeviceShaderDrawParameterFeatures>(|x| {
            &mut x.shader_draw_parameters
        })
        .unwrap();

        // Add egui plugin
        app.add_plugins(pumicite_egui::EguiPlugin::<With<PrimaryWindow>>::default());

        // Systems
        app.add_systems(Startup, setup);
        app.add_systems(EguiPrimaryContextPass, egui_ui);
        app.add_systems(
            PostUpdate,
            (prepare_atmosphere_uniform, compute_luts, render_skyview_lut)
                .chain()
                .in_set(SkyAtmosphereLUTRenderSet),
        );
    }
}

#[derive(SystemSet, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy, Debug)]
pub struct SkyAtmosphereLUTRenderSet;

// Atmosphere parameters - must match shader struct layout exactly
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod, Debug)]
struct AtmosphereParams {
    // Planet geometry (km)
    bottom_radius: f32,
    top_radius: f32,
    _pad0: [f32; 2],

    // Rayleigh scattering
    rayleigh_scattering: [f32; 3],
    rayleigh_density_exp_scale: f32,

    // Mie scattering
    mie_scattering: [f32; 3],
    mie_density_exp_scale: f32,

    mie_extinction: [f32; 3],
    mie_phase_g: f32,

    mie_absorption: [f32; 3],
    _pad1: f32,

    // Ozone absorption
    absorption_extinction: [f32; 3],
    absorption_density_0_layer_width: f32,

    absorption_density_0_constant_term: f32,
    absorption_density_0_linear_term: f32,
    absorption_density_1_constant_term: f32,
    absorption_density_1_linear_term: f32,

    // Ground
    ground_albedo: [f32; 3],
    _pad2: f32,

    // Sun
    sun_direction: [f32; 3],
    sun_angular_radius: f32,

    solar_irradiance: [f32; 3],
    _pad3: f32,

    // Camera
    camera_position: [f32; 3],
    _pad4: f32,

    // View matrices
    inv_view_proj_mat: [[f32; 4]; 4],
    inv_proj_mat: [[f32; 4]; 4],
    inv_view_mat: [[f32; 4]; 4],

    // Resolution
    resolution: [f32; 2],
    _pad5: [f32; 2],
}

impl Default for AtmosphereParams {
    fn default() -> Self {
        Self::earth_default()
    }
}

impl AtmosphereParams {
    /// Earth-like atmosphere defaults
    fn earth_default() -> Self {
        Self {
            bottom_radius: 6360.0,
            top_radius: 6460.0,
            _pad0: [0.0; 2],

            // Rayleigh scattering at sea level (1/km) - blue sky
            rayleigh_scattering: [0.005802, 0.013558, 0.033100],
            rayleigh_density_exp_scale: -1.0 / 8.0,

            // Mie scattering at sea level (1/km) - haze
            mie_scattering: [0.003996, 0.003996, 0.003996],
            mie_density_exp_scale: -1.0 / 1.2,

            mie_extinction: [0.004440, 0.004440, 0.004440],
            mie_phase_g: 0.8,

            mie_absorption: [0.000444, 0.000444, 0.000444],
            _pad1: 0.0,

            // Ozone absorption (1/km)
            absorption_extinction: [0.000650, 0.001881, 0.000085],
            absorption_density_0_layer_width: 25.0,
            absorption_density_0_constant_term: -2.0 / 3.0,
            absorption_density_0_linear_term: 1.0 / 15.0,
            absorption_density_1_constant_term: 8.0 / 3.0,
            absorption_density_1_linear_term: -1.0 / 15.0,

            ground_albedo: [0.3, 0.3, 0.3],
            _pad2: 0.0,

            sun_direction: [0.0, 0.4, 0.9165],
            sun_angular_radius: 0.00935 / 2.0,

            solar_irradiance: [1.0, 1.0, 1.0],
            _pad3: 0.0,

            camera_position: [0.0, 0.5, -0.3363016], // Just above ground
            _pad4: 0.0,

            inv_view_proj_mat: [
                [0.0, -0.0, 1.0264004, -0.0],
                [-0.0, 0.5773502, -0.0, 0.0],
                [0.0, 499.99994, -336.30154, 999.9999],
                [0.99999994, 0.0, -0.0, 0.0],
            ],
            inv_proj_mat: [
                [1.0264004, -0.0, 0.0, -0.0],
                [-0.0, 0.5773502, -0.0, 0.0],
                [0.0, -0.0, 0.0, 999.9999],
                [-0.0, 0.0, -0.99999994, 0.0],
            ],
            inv_view_mat: [
                [0.0, -0.0, 1.0, -0.0],
                [-0.0, 1.0, -0.0, 0.0],
                [-1.0, -0.0, 0.0, -0.0],
                [-0.0, 0.5, -0.3363016, 1.0],
            ],

            resolution: [2560.0, 1440.0],
            _pad5: [0.0; 2],
        }
    }
}

#[derive(Resource)]
pub struct AtmosphereState {
    params: AtmosphereParams,
    sun_elevation: f32, // radians
    sun_azimuth: f32,   // radians
    needs_lut_update: bool,

    pub uniform_buffer: Option<RingBufferSuballocation>,
}

impl Default for AtmosphereState {
    fn default() -> Self {
        Self {
            params: AtmosphereParams::earth_default(),
            sun_elevation: 0.4,
            sun_azimuth: 0.0,
            needs_lut_update: true,
            uniform_buffer: None,
        }
    }
}

#[derive(Resource)]
struct Pipelines {
    transmittance_lut: Handle<ComputePipeline>,
    multi_scattering: Handle<ComputePipeline>,
    sky_view_lut: Handle<ComputePipeline>,
}

pub struct LutImage {
    pub view: GPUMutex<FullImageView<Image>>,
    pub state: ResourceState,
}

#[derive(Resource)]
pub struct AtmosphereLUTs {
    pub transmittance: LutImage,
    multi_scattering: LutImage,
    pub sky_view: LutImage,
    pub sampler: Sampler,
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>, allocator: Res<Allocator>) {
    // Load pipelines
    commands.insert_resource(Pipelines {
        transmittance_lut: asset_server
            .load("bazel://crates/pbr/shaders/sky_atmosphere:transmittance_lut.comp.pipeline.ron"),
        multi_scattering: asset_server
            .load("bazel://crates/pbr/shaders/sky_atmosphere:multi_scattering.comp.pipeline.ron"),
        sky_view_lut: asset_server
            .load("bazel://crates/pbr/shaders/sky_atmosphere:sky_view_lut.comp.pipeline.ron"),
    });

    // Create LUT images
    let transmittance = create_lut_image(
        &allocator,
        TRANSMITTANCE_WIDTH,
        TRANSMITTANCE_HEIGHT,
        vk::Format::R16G16B16A16_SFLOAT,
        c"Atmospheric Transmittance",
        c"Atmospheric Transmittance View",
    );
    let multi_scattering = create_lut_image(
        &allocator,
        MULTI_SCATTERING_RES,
        MULTI_SCATTERING_RES,
        vk::Format::R16G16B16A16_SFLOAT,
        c"Atmospheric Multi-Scattering",
        c"Atmospheric Multi-Scattering View",
    );
    let sky_view = create_lut_image(
        &allocator,
        SKY_VIEW_WIDTH,
        SKY_VIEW_HEIGHT,
        vk::Format::B10G11R11_UFLOAT_PACK32,
        c"Atmospheric Sky-View",
        c"Atmospheric Sky-View View",
    );

    // Create sampler
    let sampler = Sampler::new(
        allocator.device().clone(),
        &vk::SamplerCreateInfo {
            mag_filter: vk::Filter::LINEAR,
            min_filter: vk::Filter::LINEAR,
            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
            address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            ..Default::default()
        },
    )
    .unwrap();

    commands.insert_resource(AtmosphereLUTs {
        transmittance,
        multi_scattering,
        sky_view,
        sampler,
    });

    commands.insert_resource(AtmosphereState::default());
}

fn create_lut_image(
    allocator: &Allocator,
    width: u32,
    height: u32,
    format: vk::Format,
    name_image: &CStr,
    name_view: &CStr,
) -> LutImage {
    let image = Image::new_private(
        allocator.clone(),
        &vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format,
            extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        },
    )
    .unwrap()
    .with_name(name_image);

    let view = GPUMutex::new(image.create_full_view().unwrap().with_name(name_view));

    LutImage {
        view,
        state: Default::default(),
    }
}

fn egui_ui(mut contexts: EguiContexts, mut atmosphere: ResMut<AtmosphereState>) {
    egui::Window::new("Sky Atmosphere")
        .default_width(300.0)
        .show(contexts.ctx_mut().unwrap(), |ui| {
            ui.heading("Sun");
            if ui
                .add(
                    egui::Slider::new(&mut atmosphere.sun_elevation, -0.5..=1.57).text("Elevation"),
                )
                .changed()
            {
                atmosphere.needs_lut_update = true;
            }
            ui.add(egui::Slider::new(&mut atmosphere.sun_azimuth, -3.14..=3.14).text("Azimuth"));

            atmosphere.params.sun_direction = [
                atmosphere.sun_azimuth.cos() * atmosphere.sun_elevation.cos(),
                atmosphere.sun_elevation.sin(),
                atmosphere.sun_azimuth.sin() * atmosphere.sun_elevation.cos(),
            ];

            ui.separator();
            ui.heading("Rayleigh Scattering");

            let base_rayleigh = [0.005802f32, 0.013558, 0.033100];
            let mut rayleigh_factor = atmosphere.params.rayleigh_scattering[0] / base_rayleigh[0];
            if ui
                .add(egui::Slider::new(&mut rayleigh_factor, 0.0..=2.0).text("Factor"))
                .changed()
            {
                atmosphere.params.rayleigh_scattering = [
                    base_rayleigh[0] * rayleigh_factor,
                    base_rayleigh[1] * rayleigh_factor,
                    base_rayleigh[2] * rayleigh_factor,
                ];
                atmosphere.needs_lut_update = true;
            }

            ui.separator();
            ui.heading("Mie Scattering");

            let base_mie_scattering = 0.003996f32;
            let base_mie_extinction = 0.004440f32;
            let mut mie_factor = atmosphere.params.mie_scattering[0] / base_mie_scattering;
            if ui
                .add(egui::Slider::new(&mut mie_factor, 0.0..=2.0).text("Factor"))
                .changed()
            {
                atmosphere.params.mie_scattering = [base_mie_scattering * mie_factor; 3];
                atmosphere.params.mie_extinction = [base_mie_extinction * mie_factor; 3];
                atmosphere.needs_lut_update = true;
            }

            if ui
                .add(
                    egui::Slider::new(&mut atmosphere.params.mie_phase_g, 0.0..=0.999)
                        .text("Phase G"),
                )
                .changed()
            {
                atmosphere.needs_lut_update = true;
            }

            ui.separator();
            ui.heading("Ground");

            let mut albedo = atmosphere.params.ground_albedo[0];
            if ui
                .add(egui::Slider::new(&mut albedo, 0.0..=1.0).text("Albedo"))
                .changed()
            {
                atmosphere.params.ground_albedo = [albedo; 3];
                atmosphere.needs_lut_update = true;
            }

            ui.separator();
            if ui.button("Reset to Earth Defaults").clicked() {
                atmosphere.params = AtmosphereParams::earth_default();
                atmosphere.sun_elevation = 0.4;
                atmosphere.sun_azimuth = 0.0;
                atmosphere.needs_lut_update = true;
            }

            ui.separator();
            ui.label("Controls:");
            ui.label("WASD - Move camera");
            ui.label("Right mouse - Look around");
            ui.label("Arrow keys - Sun position");
            ui.label("Shift - Move faster");
        });
}

fn prepare_atmosphere_uniform(
    mut ring_buffer: ResMut<UniformRingBuffer>,
    mut atmosphere: ResMut<AtmosphereState>,
) {
    // Allocate uniform buffer
    let mut buffer =
        ring_buffer.allocate_buffer(std::mem::size_of::<AtmosphereParams>() as u64, 256);
    buffer
        .as_slice_mut()
        .unwrap()
        .copy_from_slice(bytemuck::bytes_of(&atmosphere.params));

    atmosphere.uniform_buffer = Some(buffer);
}

fn compute_luts(
    atmosphere: Res<AtmosphereState>,
    mut luts: ResMut<AtmosphereLUTs>,
    pipelines: Res<Pipelines>,
    compute_pipelines: Res<Assets<ComputePipeline>>,
    mut state: SubmissionState,
) {
    if !atmosphere.needs_lut_update {
        return;
    }

    let Some(transmittance_pipeline) = compute_pipelines.get(&pipelines.transmittance_lut) else {
        return;
    };
    let Some(multi_scattering_pipeline) = compute_pipelines.get(&pipelines.multi_scattering) else {
        return;
    };

    //atmosphere.needs_lut_update = false;
    let atmosphere_uniform_buffer = atmosphere.uniform_buffer.as_ref().unwrap().clone();

    state.record(|encoder| {
        let buffer = encoder.retain(atmosphere_uniform_buffer);

        let buffer_info = vk::DescriptorBufferInfo {
            buffer: buffer.vk_handle(),
            offset: buffer.offset(),
            range: buffer.size(),
        };

        // Transmittance LUT
        {
            let transmittance_view = encoder.lock(
                &luts.transmittance.view,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
            );

            encoder.use_image_resource(
                transmittance_view.image(),
                &mut luts.transmittance.state,
                Access::COMPUTE_WRITE,
                vk::ImageLayout::GENERAL,
                0..1,
                0..1,
                true,
            );
            encoder.emit_barriers();

            let pipeline = encoder.retain(transmittance_pipeline.clone().into_inner());
            encoder.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &pipeline);

            let image_info = vk::DescriptorImageInfo {
                image_view: transmittance_view.vk_handle(),
                image_layout: vk::ImageLayout::GENERAL,
                sampler: vk::Sampler::null(),
            };

            encoder.push_descriptor_set(
                vk::PipelineBindPoint::COMPUTE,
                pipeline.layout(),
                0,
                &[
                    vk::WriteDescriptorSet {
                        dst_binding: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                        p_buffer_info: &buffer_info,
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_binding: 1,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        p_image_info: &image_info,
                        ..Default::default()
                    },
                ],
            );

            encoder.dispatch(UVec3::new(
                TRANSMITTANCE_WIDTH.div_ceil(8),
                TRANSMITTANCE_HEIGHT.div_ceil(8),
                1,
            ));
        }

        // Multi-scattering LUT (depends on transmittance)
        {
            let transmittance_view = encoder.lock(
                &luts.transmittance.view,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
            );
            let multi_scattering_view = encoder.lock(
                &luts.multi_scattering.view,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
            );

            encoder.use_image_resource(
                transmittance_view.image(),
                &mut luts.transmittance.state,
                Access::COMPUTE_READ,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                0..1,
                0..1,
                false,
            );
            encoder.use_image_resource(
                multi_scattering_view.image(),
                &mut luts.multi_scattering.state,
                Access::COMPUTE_WRITE,
                vk::ImageLayout::GENERAL,
                0..1,
                0..1,
                true,
            );
            encoder.emit_barriers();

            let pipeline = encoder.retain(multi_scattering_pipeline.clone().into_inner());
            encoder.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &pipeline);

            let transmittance_image_info = vk::DescriptorImageInfo {
                image_view: transmittance_view.vk_handle(),
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                sampler: vk::Sampler::null(),
            };

            let sampler_info = vk::DescriptorImageInfo {
                sampler: luts.sampler.vk_handle(),
                ..Default::default()
            };

            let multi_scattering_image_info = vk::DescriptorImageInfo {
                image_view: multi_scattering_view.vk_handle(),
                image_layout: vk::ImageLayout::GENERAL,
                sampler: vk::Sampler::null(),
            };

            encoder.push_descriptor_set(
                vk::PipelineBindPoint::COMPUTE,
                pipeline.layout(),
                0,
                &[
                    vk::WriteDescriptorSet {
                        dst_binding: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                        p_buffer_info: &buffer_info,
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_binding: 1,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                        p_image_info: &transmittance_image_info,
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_binding: 2,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::SAMPLER,
                        p_image_info: &sampler_info,
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_binding: 3,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        p_image_info: &multi_scattering_image_info,
                        ..Default::default()
                    },
                ],
            );

            // Dispatch with 64 threads in z for sphere integration
            encoder.dispatch(UVec3::new(MULTI_SCATTERING_RES, MULTI_SCATTERING_RES, 1));
        }
    });
}

/// Writes into [`AtmosphereLUTs::sky_view`]
fn render_skyview_lut(
    atmosphere: Res<AtmosphereState>,
    mut luts: ResMut<AtmosphereLUTs>,
    pipelines: Res<Pipelines>,
    compute_pipelines: Res<Assets<ComputePipeline>>,
    mut state: SubmissionState,
) {
    let Some(sky_view_pipeline) = compute_pipelines.get(&pipelines.sky_view_lut) else {
        return;
    };
    let atmosphere_uniform_buffer = atmosphere.uniform_buffer.as_ref().unwrap().clone();

    state.record(|encoder| {
        let buffer = encoder.retain(atmosphere_uniform_buffer);

        let buffer_info = vk::DescriptorBufferInfo {
            buffer: buffer.vk_handle(),
            offset: buffer.offset(),
            range: buffer.size(),
        };

        // Sky View LUT (per-frame)
        {
            let transmittance_view = encoder.lock(
                &luts.transmittance.view,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
            );
            let multi_scattering_view = encoder.lock(
                &luts.multi_scattering.view,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
            );
            let sky_view =
                encoder.lock(&luts.sky_view.view, vk::PipelineStageFlags2::COMPUTE_SHADER);

            encoder.use_image_resource(
                transmittance_view.image(),
                &mut luts.transmittance.state,
                Access::COMPUTE_READ,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                0..1,
                0..1,
                false,
            );
            encoder.use_image_resource(
                multi_scattering_view.image(),
                &mut luts.multi_scattering.state,
                Access::COMPUTE_READ,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                0..1,
                0..1,
                false,
            );
            encoder.use_image_resource(
                sky_view.image(),
                &mut luts.sky_view.state,
                Access::COMPUTE_WRITE,
                vk::ImageLayout::GENERAL,
                0..1,
                0..1,
                true,
            );
            encoder.emit_barriers();

            let pipeline = encoder.retain(sky_view_pipeline.clone().into_inner());
            encoder.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &pipeline);

            let transmittance_image_info = vk::DescriptorImageInfo {
                image_view: transmittance_view.vk_handle(),
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                sampler: vk::Sampler::null(),
            };
            let multi_scattering_image_info = vk::DescriptorImageInfo {
                image_view: multi_scattering_view.vk_handle(),
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                sampler: vk::Sampler::null(),
            };
            let sampler_info = vk::DescriptorImageInfo {
                sampler: luts.sampler.vk_handle(),
                ..Default::default()
            };
            let sky_view_image_info = vk::DescriptorImageInfo {
                image_view: sky_view.vk_handle(),
                image_layout: vk::ImageLayout::GENERAL,
                sampler: vk::Sampler::null(),
            };

            encoder.push_descriptor_set(
                vk::PipelineBindPoint::COMPUTE,
                pipeline.layout(),
                0,
                &[
                    vk::WriteDescriptorSet {
                        dst_binding: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                        p_buffer_info: &buffer_info,
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_binding: 1,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                        p_image_info: &transmittance_image_info,
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_binding: 2,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                        p_image_info: &multi_scattering_image_info,
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_binding: 3,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::SAMPLER,
                        p_image_info: &sampler_info,
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_binding: 4,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        p_image_info: &sky_view_image_info,
                        ..Default::default()
                    },
                ],
            );

            encoder.dispatch(UVec3::new(
                SKY_VIEW_WIDTH.div_ceil(8),
                SKY_VIEW_HEIGHT.div_ceil(8),
                1,
            ));
        }
    });
}
