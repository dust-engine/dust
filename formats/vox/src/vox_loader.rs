use std::sync::Arc;

use ash::vk;
use bevy_ecs::system::lifetimeless::SRes;
use dot_vox::Model;
use dust_render::render_asset::{RenderAsset, GPURenderAsset, GPURenderAssetBuildResult};
use dust_vdb::hierarchy;
use dustash::resources::alloc::{Allocator, MemBuffer, BufferRequest, MemoryAllocScenario};
use glam::{UVec3, Vec3};
/// MagicaVoxel trees are 256x256x256 max, so the numbers in the
/// hierarchy must sum up to 8 where 2^8 = 256.

pub type TreeRoot = hierarchy!(4, 2, 2);
pub type Tree = dust_vdb::Tree<TreeRoot>;
use crate::palette::{Color, VoxPalette};



//  2,266,302 ns/iter
pub fn convert_model(allocator: &Arc<Allocator>, model: &Model, palette: &Handle<VoxPalette>) -> (Tree, PaletteMaterial) {
    let mut palette_index_collector = crate::collector::ModelIndexCollector::new();


    let mut tree = Tree::new();
    for voxel in model.voxels.iter() {
        let coords: UVec3 = UVec3 {
            x: voxel.x as u32,
            y: voxel.z as u32,
            z: voxel.y as u32,
        };
        tree.set_value(coords, Some(true));
        palette_index_collector.set(voxel.clone());
    }

    let palette_indexes = palette_index_collector.into_iter();
    // TODO: use iter_leaf_mut here, and insert indices
    for (location, leaf) in tree.iter_leaf_mut() {
        let block_index = (location.x >> 2, location.y >> 2, location.z >> 2);
        let block_index = block_index.0 as usize + block_index.1 as usize * 64 + block_index.2 as usize * 64 * 64;

        leaf.material_ptr = palette_indexes.running_sum()[block_index];
    }

    use dust_render::attributes::AttributeWriter;
    let writer = dust_render::attributes::IntegerWriter::from_iter(allocator, palette_indexes);
    (
        tree,
        PaletteMaterial::new(palette.clone(), writer.into_resource())
    )
    
}

use bevy_asset::{AssetLoader, LoadedAsset, Handle};

use crate::material::PaletteMaterial;

pub struct VoxLoader {
    allocator: Arc<Allocator>,
}

impl VoxLoader {
    pub fn new(allocator: Arc<Allocator>) -> Self {
        Self {
            allocator
        }
    }
}

impl AssetLoader for VoxLoader {
    fn load<'a>(
        &'a self,
        bytes: &'a [u8],
        load_context: &'a mut bevy_asset::LoadContext,
    ) -> bevy_asset::BoxedFuture<'a, Result<(), anyhow::Error>> {
        Box::pin(async {
            let file = dot_vox::load_bytes(bytes).map_err(|str| anyhow::Error::msg(str))?;

            let palette = unsafe {
                const LEN: usize = 256;
                let mem = std::alloc::alloc(std::alloc::Layout::new::<[Color; LEN]>()) as *mut [Color; LEN];
                let mut mem = Box::from_raw(mem);
                assert_eq!(file.palette.len(), 256);
                for i in 0..256 {
                    mem[i] = std::mem::transmute(file.palette[i]);
                }
                println!("Palette is {:?}", mem);
                VoxPalette(mem)
            };
            let palette_handle = load_context.set_labeled_asset("palette", LoadedAsset::new(palette));


            let (i, model) = file.models.iter().enumerate().max_by_key(|a| a.1.voxels.len()).unwrap();
            let (tree, material) = convert_model(&self.allocator, model, &palette_handle);
            let geometry = crate::VoxGeometry::from_tree(tree, 1.0);

            println!("Asset loaded {}", i);
            load_context.set_default_asset(LoadedAsset::new(geometry));

            load_context.set_labeled_asset("material", LoadedAsset::new(material));
            Ok(())
        })
    }

    fn extensions(&self) -> &[&str] {
        &["vox"]
    }
}
