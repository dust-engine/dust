use dot_vox::Model;
use dust_vdb::hierarchy;
use glam::UVec3;
/// MagicaVoxel trees are 256x256x256 max, so the numbers in the
/// hierarchy must sum up to 8 where 2^8 = 256.
type Tree = dust_vdb::Tree<hierarchy!(4, 2, 2)>;

//  2,266,302 ns/iter
pub fn convert_model(model: &Model) -> Tree {
    let mut tree = Tree::new();
    for voxel in model.voxels.iter() {
        let coords: UVec3 = UVec3 {
            x: voxel.x as u32,
            y: voxel.y as u32,
            z: voxel.z as u32,
        };
        for i in 0..8 {
            tree.set_value(coords, Some(true));
        }
    }
    // next step should be surface extraction
    tree
}
