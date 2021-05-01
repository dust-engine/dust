use crate::{Voxel, Corner};
use super::{Node, Octree, accessor::tree::NodeRefMut};
use crate::alloc::{ArenaAllocator};
use glam::Vec3;

impl<T: Voxel> Octree<T>
{
    fn signed_distance_field_recursive<F>(
        signed_distance_field: &F,
        fill: T,
        lod: u8,
        mut node: NodeRefMut<T>,
    ) where
        F: Fn(Vec3) -> f32,
    {
        assert!(!node.is_virtual()); // Can't be a virtual node
        let mut childmask: u8 = 0;
        for (i, dir) in Corner::all().enumerate() {
            let child = node.child(dir);
            let mut corners_flag: u8 = 0;
            for (i, dir) in Corner::all().enumerate() {
                let corner = child.get_bounds().corner(dir);
                let value = signed_distance_field(corner);
                if value > 0.0 {
                    corners_flag |= 1 << i;
                }
            }
            if corners_flag == 0 {
                // All corners smaller than zero. Do not fill.
            } else if corners_flag == std::u8::MAX {
                // All corners larger than zero. Fill.
                node.set_leaf_child(dir, fill);
            } else {
                // This child node needs to be subdivided
                childmask |= 1 << i;
                if corners_flag.count_ones() >= 4 {
                    // relative full
                    node.set_leaf_child(dir, fill);
                }
                // Set the childmask here so it can be handled later on.
            }
        }
        if lod > 0 {
            node.set_leaf_childmask(childmask);

            for (i, dir) in Corner::all().enumerate() {
                if (1 << i) & childmask == 0 {
                    // No child on this corner
                    continue;
                }
                let child = node.child(dir);
                Octree::signed_distance_field_recursive(
                    signed_distance_field,
                    fill,
                    lod - 1,
                    child,
                );
            }
        }
    }
    pub fn from_signed_distance_field<F>(&mut self, field: F, fill: T, lod: u8)
        where
            F: Fn(Vec3) -> f32,
    {
        Octree::signed_distance_field_recursive(&field, fill, lod, self.get_tree_mutator());
    }
}