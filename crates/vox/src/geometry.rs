use std::sync::Arc;

use crate::{Tree, TreeRoot, VoxPalette};

use bevy_asset::Asset;
use dust_render::Geometry;
use dust_vdb::{IsLeaf, Node};
use glam::{UVec3, Vec3A, Vec4};
use rhyolite::ash::vk;
use rhyolite::debug::DebugObject;
use rhyolite::future::{GPUCommandFuture, GPUCommandFutureExt, UnitCommandFuture};
use rhyolite::ResidentBuffer;
use rhyolite_bevy::{Allocator, StagingRingBuffer};

#[derive(bevy_reflect::TypePath, Asset)]
pub struct VoxGeometry {
    tree: Tree,
    size: [u8; 3],
    pub num_blocks: u32,
    pub unit_size: f32,

    /// Array of AABBs, used as Acceleration Strucutre Build Input
    aabb_buffer: Arc<ResidentBuffer>,

    /// Array of `GPUVoxNode`, used during ray tracing.
    /// Its shader device address is written into the SBT Records
    geometry_buffer: Arc<ResidentBuffer>,
}

impl Geometry for VoxGeometry {
    const TYPE: dust_render::GeometryType = dust_render::GeometryType::AABBs;

    type BLASInputBufferFuture = UnitCommandFuture<Arc<ResidentBuffer>>;

    fn blas_input_buffer(&self) -> Self::BLASInputBufferFuture {
        UnitCommandFuture::new(self.aabb_buffer.clone())
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

impl VoxGeometry {
    pub fn geometry_buffer(&self) -> &Arc<ResidentBuffer> {
        &self.geometry_buffer
    }
    pub fn from_tree(
        tree: Tree,
        size: [u8; 3],
        unit_size: f32,
        allocator: &Allocator,
        ring_buffer: &StagingRingBuffer,
        palette_indexes: &[u8],
        palette: &VoxPalette
    ) -> impl GPUCommandFuture<Output = Self> {
        let leaf_extent_int = <<TreeRoot as Node>::LeafType as Node>::EXTENT;
        let leaf_extent: Vec3A = leaf_extent_int.as_vec3a();
        let leaf_extent: Vec3A = unit_size * leaf_extent;

        let (aabbs, nodes): (Vec<vk::AabbPositionsKHR>, Vec<GPUVoxNode>) = tree
            .iter_leaf()
            .map(|(position, d)| {
                let aabb = {
                    let position = position.as_vec3a();
                    let max_position = leaf_extent + position;
                    vk::AabbPositionsKHR {
                        min_x: position.x,
                        min_y: position.y,
                        min_z: position.z,
                        max_x: max_position.x,
                        max_y: max_position.y,
                        max_z: max_position.z,
                    }
                };
                let mut mask = [0_u64; 1];
                d.get_occupancy(&mut mask);

                let mut color = glam::UVec4::ZERO;
                let num_voxels = mask[0].count_ones();
                for i in 0..num_voxels {
                    let palette_index = palette_indexes[d.material_ptr as usize + i as usize];
                    let albedo = palette.colors[palette_index as usize];
                    color += glam::UVec4::new(
                        albedo.r as u32,
                        albedo.g as u32,
                        albedo.b as u32,
                        albedo.a as u32,
                    );
                }
                let mut color = color.as_vec4() / (num_voxels as f32 * 255.0);
                fn linear2srgb(color: f32) -> f32{
                    if color <= 0.0031308 { 12.92 * color } else { 1.055 * color.powf(1.0 / 2.4) - 0.055 }
                }
                color.x = linear2srgb(color.x);
                color.y = linear2srgb(color.y);
                color.z = linear2srgb(color.z);
                let r = (color.x * 1023.0) as u32;
                let g = (color.y * 1023.0) as u32;
                let b = (color.z * 1023.0) as u32;
                let a = (color.w * 3.0) as u32;
                let packed = (r << 22) | (g << 12) | (b << 2) | a;

                let node = {
                    GPUVoxNode {
                        x: position.x as u16,
                        y: position.y as u16,
                        z: position.z as u16,
                        w: 0,
                        mask: mask[0],
                        material_ptr: d.material_ptr,
                        avg_albedo: packed,
                    }
                };
                (aabb, node)
            })
            .unzip();
        let aabb_buffer = {
            let size = std::mem::size_of_val(aabbs.as_slice());
            assert_eq!(size, aabbs.len() * 24);
            let data = unsafe { std::slice::from_raw_parts(aabbs.as_ptr() as *const u8, size) };
            allocator
                .create_static_device_buffer_with_data(
                    data,
                    vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                        | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                    16,
                    &ring_buffer,
                )
                .unwrap()
                .map(|buffer| {
                    buffer.inspect(|buffer| {
                        buffer.set_name("Vox BLAS Input AABB Buffer").unwrap();
                    })
                })
        };
        let geometry_buffer = {
            let size = std::mem::size_of_val(nodes.as_slice());
            assert_eq!(size, nodes.len() * 24);
            let data = unsafe { std::slice::from_raw_parts(nodes.as_ptr() as *const u8, size) };
            allocator
                .create_static_device_buffer_with_data(
                    data,
                    vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    16,
                    &ring_buffer,
                )
                .unwrap()
                .map(|buffer| {
                    buffer.inspect(|buffer| {
                        buffer.set_name("Vox Geometry Buffer").unwrap();
                    })
                })
        };
        let num_blocks = aabbs.len() as u32;
        let future =
            aabb_buffer
                .join(geometry_buffer)
                .map(move |(aabb_buffer, geometry_buffer)| Self {
                    tree,
                    size,
                    unit_size,
                    aabb_buffer: Arc::new(aabb_buffer.into_inner()),
                    geometry_buffer: Arc::new(geometry_buffer.into_inner()),
                    num_blocks,
                });
        future
    }
    pub fn set(&mut self, coords: UVec3, value: Option<bool>) {
        self.tree.set_value(coords, value)
    }
    pub fn get(&mut self, coords: UVec3) -> Option<bool> {
        self.tree.get_value(coords)
    }
}
