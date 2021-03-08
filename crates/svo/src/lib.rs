#![feature(untagged_unions)]
#![feature(const_fn)]

mod arena;
pub mod bounds;
pub mod dir;
pub mod index_path;
pub mod octree;

pub use arena::{Arena, ArenaHandle};
pub use bounds::Bounds;
pub use dir::{Corner, Edge, Face, Quadrant};
pub use index_path::IndexPath;
pub use octree::{NodeRef, NodeRefMut, Octree};

use std::fmt::Debug;

pub trait Voxel: Copy + Clone + Default + Eq + Debug {
    fn avg(voxels: &[Self; 8]) -> Self;
}

#[cfg(test)]
mod tests {
    use super::*;
    impl Voxel for u32 {
        fn avg(arr: &[Self; 8]) -> Self {
            // find most frequent element
            let mut arr = arr.clone();
            arr.sort();

            let mut count: u8 = 0;
            let mut max_count: u8 = 0;
            let mut max_element: Self = 0;
            let mut last_element: Self = 0;
            for i in &arr {
                if *i != last_element {
                    if count > max_count {
                        max_count = count;
                        max_element = *i;
                    }
                    count = 0;
                }
                count += 1;
                last_element = *i;
            }
            max_element
        }
    }
    impl Voxel for u16 {
        fn avg(arr: &[Self; 8]) -> Self {
            // find most frequent element
            let mut arr = arr.clone();
            arr.sort();

            let mut count: u8 = 0;
            let mut max_count: u8 = 0;
            let mut max_element: Self = 0;
            let mut last_element: Self = 0;
            for i in &arr {
                if *i != last_element {
                    if count > max_count {
                        max_count = count;
                        max_element = *i;
                    }
                    count = 0;
                }
                count += 1;
                last_element = *i;
            }
            max_element
        }
    }
}
