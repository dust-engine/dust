use dot_vox::Model;
use dust_format_vdb::hierarchy;
use glam::{UVec3, Vec3};
/// MagicaVoxel trees are 256x256x256 max, so the numbers in the
/// hierarchy must sum up to 8 where 2^8 = 256.

pub type TreeRoot = hierarchy!(4, 2, 2);
pub type Tree = dust_format_vdb::Tree<TreeRoot>;

//  2,266,302 ns/iter
pub fn convert_model(model: &Model) -> Tree {
    let mut tree = Tree::new();
    for voxel in model.voxels.iter() {
        let coords: UVec3 = UVec3 {
            x: voxel.x as u32,
            y: voxel.y as u32,
            z: voxel.z as u32,
        };
        tree.set_value(coords, Some(true));
    }
    // next step should be surface extraction
    tree
}

use bevy_asset::{AssetLoader, LoadedAsset};

#[derive(Default)]
pub struct VoxLoader;
impl AssetLoader for VoxLoader {
    fn load<'a>(
        &'a self,
        bytes: &'a [u8],
        load_context: &'a mut bevy_asset::LoadContext,
    ) -> bevy_asset::BoxedFuture<'a, Result<(), anyhow::Error>> {
        Box::pin(async {
            let model = dot_vox::load_bytes(bytes).map_err(|str| anyhow::Error::msg(str))?;
            let model = model.models.iter().max_by_key(|a| a.voxels.len()).unwrap();
            let tree = convert_model(model);
            let geometry = dust_format_vdb::VdbGeometry::new(tree, Vec3::ONE);
            load_context.set_default_asset(LoadedAsset::new(geometry));
            Ok(())
        })
    }

    fn extensions(&self) -> &[&str] {
        &["aabb"]
    }
}
