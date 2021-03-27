use super::super::Octree;
use crate::alloc::CHUNK_SIZE;
use crate::octree::Node;
use crate::Voxel;
use std::mem::size_of;

pub struct RandomAccessor<'a, T: Voxel> {
    octree: &'a Octree<T>,
}
