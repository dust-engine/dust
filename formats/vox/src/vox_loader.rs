use dot_vox::Model;
use dust_vdb::hierarchy;
use glam::UVec3;
/// MagicaVoxel trees are 256x256x256 max, so the numbers in the
/// hierarchy must sum up to 8 where 2^8 = 256.
type Tree = dust_vdb::Tree<hierarchy!(4, 2, 2)>;

//  3,189,600 ns/iter
//  2,417,170 ns/iter
pub fn convert_model(model: &Model) -> Tree {
    let mut tree = Tree::new();
    for voxel in model.voxels.iter() {
        let coords: UVec3 = UVec3 {
            x: voxel.x as u32,
            y: voxel.y as u32,
            z: voxel.z as u32,
        };
        tree.set_value(coords, Some(false));
    }
    tree
}


// 10,099,750 ns/iter
// 7,222,370 ns/iter
pub fn convert_model2(model: &Model) -> Tree {
    let mut tree = Tree::new();
    let mut accessor = tree.accessor_mut();
    for voxel in model.voxels.iter() {
        let coords: UVec3 = UVec3 {
            x: voxel.x as u32,
            y: voxel.y as u32,
            z: voxel.z as u32,
        };
        accessor.set(coords, Some(false));
    }
    tree
}




#[cfg(test)]
mod tests {
    extern crate test;
    use test::Bencher;
    #[bench]
    fn run_test(bencher: &mut Bencher) {
        let vox = dot_vox::load("./castle.vox").unwrap();
        bencher.iter(|| {
            let model = super::convert_model2(&vox.models[401]);
            test::black_box(model);
        });
    }
}
