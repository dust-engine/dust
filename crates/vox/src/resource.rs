use bevy::ecs::system::lifetimeless::{SRes, SResMut};
use bevy::ecs::system::SystemParamItem;
use bevy::{asset::Asset, reflect::TypePath};
use rhyolite::commands::TransferCommands;
use rhyolite::staging::StagingBelt;
use rhyolite::utils::GPUAsset;

use rhyolite::ash::vk;

use crate::{VoxGeometry, VoxMaterial, VoxPalette};
use dust_vdb::IsLeaf;
use rhyolite::{Allocator, Buffer};

#[derive(Asset, TypePath)]
pub struct VoxPaletteGPU(pub(crate) Buffer);

impl GPUAsset for VoxPaletteGPU {
    type SourceAsset = VoxPalette;

    type Params = (SRes<Allocator>, SResMut<StagingBelt>);

    fn upload_asset(
        source_asset: &Self::SourceAsset,
        commands: &mut impl TransferCommands,
        (allocator, staging_belt): &mut SystemParamItem<Self::Params>,
    ) -> Self {
        let data = unsafe {
            std::slice::from_raw_parts(
                source_asset.0.as_ptr() as *const u8,
                source_asset.0.len() * 4,
            )
        };
        let buffer = Buffer::new_resource_init(
            allocator.clone(),
            staging_belt,
            data,
            1,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            commands,
        );
        VoxPaletteGPU(buffer.unwrap())
    }
}
