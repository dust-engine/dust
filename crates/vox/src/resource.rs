use bevy::ecs::system::lifetimeless::{SRes, SResMut};
use bevy::ecs::system::SystemParamItem;
use bevy::{asset::Asset, reflect::TypePath};
use rhyolite::commands::TransferCommands;
use rhyolite::staging::StagingBelt;
use rhyolite::utils::AssetUpload;

use rhyolite::ash::vk;

use crate::{VoxGeometry, VoxMaterial, VoxPalette};
use dust_vdb::IsLeaf;
use rhyolite::{Allocator, Buffer};

#[derive(Asset, TypePath)]
pub struct VoxPaletteGPU(pub(crate) Buffer);

impl AssetUpload for VoxPalette {
    type GPUAsset = VoxPaletteGPU;

    type Params = (SRes<Allocator>, SResMut<StagingBelt>);

    fn upload_asset(
        &self,
        commands: &mut impl TransferCommands,
        (allocator, staging_belt): &mut SystemParamItem<Self::Params>,
    ) -> Self::GPUAsset {
        let data =
            unsafe { std::slice::from_raw_parts(self.0.as_ptr() as *const u8, self.0.len() * 4) };
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

#[repr(C)]
struct GPUVoxNode {
    x: u16,
    y: u16,
    z: u16,
    w: u16,
    mask: u64,
    material_ptr: u32,
    avg_albedo: u32,
}

#[derive(Asset, TypePath)]
pub struct VoxGeometryGPU(pub(crate) Buffer);
impl AssetUpload for VoxGeometry {
    type GPUAsset = VoxGeometryGPU;

    type Params = (SRes<Allocator>, SResMut<StagingBelt>);

    fn upload_asset(
        &self,
        commands: &mut impl TransferCommands,
        (allocator, staging_belt): &mut SystemParamItem<Self::Params>,
    ) -> Self::GPUAsset {
        let leaf_count = self.tree.iter_leaf().count();
        let mut current_location = 0;

        let buffer = Buffer::new_resource_init_with(
            allocator.clone(),
            staging_belt,
            leaf_count as u64 * std::mem::size_of::<GPUVoxNode>() as u64,
            1,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            commands,
            |dst| {
                for (position, d) in self.tree.iter_leaf() {
                    let mut mask = [0_u64; 1];
                    d.get_occupancy(&mut mask);
                    let node = GPUVoxNode {
                        x: position.x as u16,
                        y: position.y as u16,
                        z: position.z as u16,
                        w: 0,
                        mask: mask[0],
                        material_ptr: d.material_ptr,
                        avg_albedo: 0,
                    };
                    let dst_slice = &mut dst
                        [current_location..(current_location + std::mem::size_of::<GPUVoxNode>())];
                    dst_slice.copy_from_slice(unsafe {
                        std::slice::from_raw_parts(
                            &node as *const GPUVoxNode as *const u8,
                            std::mem::size_of::<GPUVoxNode>(),
                        )
                    });
                    current_location += std::mem::size_of::<GPUVoxNode>();
                }
            },
        )
        .unwrap();
        VoxGeometryGPU(buffer)
    }
}

#[derive(Asset, TypePath)]
pub struct VoxMaterialGPU(pub(crate) Buffer);
impl AssetUpload for VoxMaterial {
    type GPUAsset = VoxMaterialGPU;

    type Params = (SRes<Allocator>, SResMut<StagingBelt>);

    fn upload_asset(
        &self,
        commands: &mut impl TransferCommands,
        (allocator, staging_belt): &mut SystemParamItem<Self::Params>,
    ) -> Self::GPUAsset {
        let buffer = Buffer::new_resource_init(
            allocator.clone(),
            staging_belt,
            &self,
            1,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            commands,
        )
        .unwrap();
        VoxMaterialGPU(buffer)
    }
}
