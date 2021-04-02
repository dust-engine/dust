mod voxel;

pub use voxel::Voxel;
pub type Octree = svo::octree::Octree<Voxel>;

pub use svo;
