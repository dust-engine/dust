use super::super::Octree;
use crate::Voxel;
use crate::alloc::CHUNK_SIZE;
use std::mem::size_of;
use crate::octree::Node;

pub struct RandomAccessor<'a, T: Voxel>
    where [(); CHUNK_SIZE / size_of::<Node<T>>()]: Sized {
    octree: &'a Octree<T>
}