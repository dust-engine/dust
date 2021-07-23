#![allow(incomplete_features)]
#![feature(untagged_unions)]
#![feature(const_generics)]
#![feature(const_evaluatable_checked)]
#![feature(allocator_api)]
#![feature(slice_ptr_get)]
#![feature(alloc_layout_extra)]
#![feature(maybe_uninit_extra)]
#![feature(array_map)]

pub mod alloc;
pub mod bounds;
pub mod dir;
pub mod mesher;
pub mod octree;

pub use bounds::Bounds;
pub use dir::{Corner, Edge, Face, Quadrant};
pub type ArenaAllocator<T> = alloc::ArenaAllocator<octree::NodeInternal<T>>;

use std::fmt::Debug;

// Voxel must also be 2 bytes in total.
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

            let mut count: u8 = 1;
            let mut max_count: u8 = 0;
            let mut max_element: Self = 0;
            let mut last_element: Self = arr[0];
            for i in arr.iter().skip(1) {
                if *i != last_element {
                    if count > max_count {
                        max_count = count;
                        max_element = last_element;
                    }
                    count = 0;
                    last_element = *i;
                }
                count += 1;
            }
            if count > max_count {
                max_element = last_element;
            }
            max_element
        }
    }
    impl Voxel for u16 {
        fn avg(arr: &[Self; 8]) -> Self {
            // find most frequent element
            let mut arr = arr.clone();
            arr.sort();

            let mut count: u8 = 1;
            let mut max_count: u8 = 0;
            let mut max_element: Self = 0;
            let mut last_element: Self = arr[0];
            for i in arr.iter().skip(1) {
                if *i != last_element {
                    if count > max_count {
                        max_count = count;
                        max_element = last_element;
                    }
                    count = 0;
                    last_element = *i;
                }
                count += 1;
            }
            if count > max_count {
                max_element = last_element;
            }
            max_element
        }
    }
    #[test]
    fn test_voxel() {
        let arr: [u16; 8] = [0, 0, 0, 0, 1, 1, 2, 3];
        assert_eq!(Voxel::avg(&arr), 0);
        let arr: [u16; 8] = [3, 3, 3, 3, 3, 3, 0, 3];
        assert_eq!(Voxel::avg(&arr), 3);
    }
}
