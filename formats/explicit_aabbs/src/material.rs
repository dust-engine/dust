use ash::vk;
use bevy_asset::AssetLoader;
use bevy_asset::AssetServer;
use bevy_ecs::system::lifetimeless::SRes;
use dust_render::material::{GPUMaterial, Material};
use dust_render::shader::SpecializedShader;
use dustash::queue::QueueType;
use dustash::queue::Queues;
use dustash::resources::alloc::{Allocator, BufferRequest, MemBuffer, MemoryAllocScenario};
use dustash::resources::Image;
use dustash::Device;
use std::sync::Arc;

#[derive(bevy_reflect::TypeUuid)]
#[uuid = "75a9a733-04d7-4abb-8600-9a7d24ff0598"]
pub struct DensityMaterial {
    rawdata: Box<[u8]>,
    extent: vk::Extent2D,
}

impl Material for DensityMaterial {
    type Geometry = crate::AABBGeometry;

    fn anyhit_shader(asset_server: &AssetServer) -> Option<dust_render::shader::SpecializedShader> {
        None
    }

    fn closest_hit_shader(
        asset_server: &AssetServer,
    ) -> Option<dust_render::shader::SpecializedShader> {
        Some(SpecializedShader {
            shader: asset_server.load("plain.rchit.spv"),
            specialization: None,
        })
    }

    type GPUMaterial = GPUDensityMaterial;

    type ChangeSet = ();

    type BuildSet = (MemBuffer, vk::Extent2D);

    type GenerateBuildsParam = SRes<Arc<Allocator>>;

    fn generate_builds(
        &mut self,
        allocator: &mut bevy_ecs::system::SystemParamItem<Self::GenerateBuildsParam>,
    ) -> Self::BuildSet {
        let mut buffer = allocator
            .allocate_buffer(&BufferRequest {
                size: self.rawdata.len() as u64,
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                scenario: MemoryAllocScenario::StagingBuffer,
                ..Default::default()
            })
            .unwrap();
        buffer.map_scoped(|slice| {
            slice.copy_from_slice(&self.rawdata);
        });
        (buffer, self.extent)
    }

    type EmitChangesParam = ();

    fn emit_changes(
        &mut self,
        param: &mut bevy_ecs::system::SystemParamItem<Self::EmitChangesParam>,
    ) -> Self::ChangeSet {
        todo!()
    }
}

pub struct GPUDensityMaterial {
    density_map: Arc<dustash::resources::image::MemImage>,
}

impl GPUMaterial<DensityMaterial> for GPUDensityMaterial {
    fn material_info(&self) -> u64 {
        0
    }

    type BuildParam = (SRes<Arc<Device>>, SRes<Arc<Allocator>>, SRes<Arc<Queues>>);

    fn build(
        build_set: <DensityMaterial as Material>::BuildSet,
        commands_future: &mut dustash::sync::CommandsFuture,
        params: &mut bevy_ecs::system::SystemParamItem<Self::BuildParam>,
    ) -> Self {
        let (image_srcbuffer, extent) = build_set;
        let (device, allocator, queues) = params;
        let image = allocator
            .allocate_image(&dustash::resources::image::ImageRequest {
                format: vk::Format::R8_UINT,
                extent: vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                },
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST,
                initial_layout: vk::ImageLayout::UNDEFINED,
                ..Default::default()
            })
            .unwrap();
        let image = Arc::new(image);
        let queue_family_index = commands_future.queue_family_index();
        commands_future.then_commands(|mut recorder| {
            use dustash::resources::HasImage;
            recorder.pipeline_barrier(
                vk::PipelineStageFlags::empty(),
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                &[vk::ImageMemoryBarrier {
                    src_access_mask: vk::AccessFlags::NONE,
                    dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    old_layout: vk::ImageLayout::UNDEFINED,
                    new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image: image.raw_image(),
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    ..Default::default()
                }],
            );
            recorder.copy_buffer_to_image(
                image_srcbuffer,
                image.clone(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_row_length: extent.width,
                    buffer_image_height: extent.height,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    image_offset: vk::Offset3D::default(),
                    image_extent: vk::Extent3D {
                        height: extent.height,
                        width: extent.width,
                        depth: 1,
                    },
                }],
            );

            let dst_queue_family_index = queues.of_type(QueueType::Compute).family_index();

            // Queue family transfer
            recorder.pipeline_barrier(
                vk::PipelineStageFlags::TRANSFER,
                if dst_queue_family_index == queue_family_index {
                    vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR
                } else {
                    vk::PipelineStageFlags::empty()
                },
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier {
                    src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    dst_access_mask: vk::AccessFlags::SHADER_READ,
                    old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    new_layout: vk::ImageLayout::READ_ONLY_OPTIMAL,
                    src_queue_family_index: queue_family_index,
                    dst_queue_family_index,
                    image: image.raw_image(),
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    ..Default::default()
                }],
            );
        });
        Self { density_map: image }
    }

    type ApplyChangeParam = ();

    fn apply_change_set(
        &mut self,
        change_set: <DensityMaterial as Material>::ChangeSet,
        commands_future: &mut dustash::sync::CommandsFuture,
        params: &mut bevy_ecs::system::SystemParamItem<Self::ApplyChangeParam>,
    ) {
        todo!()
    }
}

#[derive(Default)]
pub struct DensityMaterialLoader;
impl AssetLoader for DensityMaterialLoader {
    fn load<'a>(
        &'a self,
        bytes: &'a [u8],
        load_context: &'a mut bevy_asset::LoadContext,
    ) -> bevy_asset::BoxedFuture<'a, Result<(), anyhow::Error>> {
        println!("Loaded Density Material File");
        Box::pin(async move {
            use bevy_asset::LoadedAsset;
            use image::ImageDecoder;
            use std::io::Cursor;
            let decoder = image::codecs::bmp::BmpDecoder::new(Cursor::new(bytes)).unwrap();
            let extent = {
                let (width, height) = decoder.dimensions();
                vk::Extent2D { width, height }
            };
            let mut rawdata: Vec<u8> = vec![0; decoder.total_bytes() as usize];
            decoder.read_image(&mut rawdata);
            let material = DensityMaterial {
                extent,
                rawdata: rawdata.into_boxed_slice(),
            };
            load_context.set_default_asset(LoadedAsset::new(material));
            Ok(())
        })
    }

    fn extensions(&self) -> &[&str] {
        &["bmp"]
    }
}
