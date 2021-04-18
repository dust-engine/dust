use crate::octree::{Octree, Node};
use crate::{Voxel, Bounds, Corner};
use crate::alloc::{ArenaAllocator, Handle, ChangeSet, BlockAllocator};
use std::collections::VecDeque;
use std::sync::Arc;

/**
A supertree is an octree of octrees.

Each octree has a dedicated ArenaAllocator backed by the same Arc<dyn BlockAllocator>.
The memory used by a single octree is therefore
A single octree is the minimal unit when doing LOD

A super tree is a 3d grid of small octrees. It'll fake an octree on top of these octrees so it's
easier to raytrace.


Coordinates:
Each region has a unique u64 coordinate. The f32 coordinates are relative to the u64 coordinates
of the region, and they can range from -1 to +1
*/
const GRID_DEGREE: u8 = 3;
const GRID_SIZE: u32 = 1 << GRID_DEGREE as u32;
const ROOT_HANDLE: Handle = Handle::from_index(0, 0);
pub trait OctreeLoader<T: Voxel> {
    fn load_region_with_lod(&self, x: u64, y: u64, z: u64, distance_to_center: u8) -> Option<Box<Octree<T>>>;
    fn unload_region(&self, x: u64, y: u64, z: u64, octree: Octree<T>);
}

pub struct Supertree<T: Voxel, L: OctreeLoader<T>> {
    loader: L,
    block_allocator: Arc<dyn BlockAllocator>,
    arena: ArenaAllocator<Node<T>>,
    // (octree, root of octree in the supertree)
    octrees: [Option<(Box<Octree<T>>, Handle)>; (GRID_SIZE * GRID_SIZE * GRID_SIZE) as usize],
    max_lod: u8,
    offset_x: u64,
    offset_y: u64,
    offset_z: u64,
}

impl<T: Voxel, L: OctreeLoader<T>> Supertree<T, L> {
    pub fn new(block_allocator: Arc<dyn BlockAllocator>, loader: L, max_lod: u8) -> Self {
        let mut arena = ArenaAllocator::new(block_allocator.clone());
        let root = arena.alloc(1);
        assert_eq!(root, ROOT_HANDLE); // root is always assumed to be at (0, 0)
        let mut tree = Supertree {
            loader,
            block_allocator,
            arena,
            octrees: unsafe { std::mem::zeroed() },
            max_lod,
            offset_x: 0,
            offset_y: 0,
            offset_z: 0
        };
        tree.load();
        tree
    }
    pub fn load(&mut self) {
        let mask = GRID_SIZE - 1; // GRID_DEGREE number of ones
        for (i, octree) in self.octrees.iter_mut().enumerate() {
            if octree.is_some() {
                continue;
            }
            let mut i = i as u32;
            let x = i & mask;
            i = i >> GRID_DEGREE;
            let y = i & mask;
            i = i >> GRID_DEGREE;
            let z = i & mask;

            fn distance(a: u32) -> u8 {
                let j = if a >= GRID_SIZE / 2 {
                    !a & (GRID_SIZE - 1)
                } else {
                    a
                };
                (GRID_SIZE / 2 - j - 1) as u8
            }

            let distance_to_center = {
                let x = distance(x);
                let y = distance(y);
                let z = distance(z);
                x.max(y).max(z)
            };
            *octree = self.loader.load_region_with_lod(
                self.offset_x + x as u64,
                self.offset_y + y as u64,
                self.offset_z + z as u64,
                distance_to_center,
            ).map(|octree| (octree, Handle::none()));
        }
        fn generate<T: Voxel>(octrees: &mut [Option<(Box<Octree<T>>, Handle)>], arena: &mut ArenaAllocator<Node<T>>, bounds: Bounds) -> Node<T> {
            // Base case:
            if bounds.width == 2 {
                let mut data: [T; 8] = Default::default();
                let mut octree_indexes: [usize; 8] = [0; 8];
                let mut freemask: u8 = 0;
                let mut num_children: u8 = 0;
                for (i, corner) in Corner::all().enumerate() {
                    let (offset_x, offset_y, offset_z) = corner.position_offset();
                    let x = bounds.x as usize + offset_x as usize;
                    let y = bounds.y as usize + offset_y as usize;
                    let z = bounds.z as usize + offset_z as usize;
                    let size = GRID_SIZE as usize;
                    let index: usize = x * size * size + y * size + z;
                    octree_indexes[i] = index;
                    if let Some(octree) = &mut octrees[index] {
                        freemask |= (1 << i);
                        data[i] = octree.0.root_data;
                        num_children += 1;
                    }
                }
                // move the root nodes together
                let child_handle = if num_children > 0 {
                    let child_handle = arena.alloc(num_children as u32);
                    arena.changed_block(child_handle, num_children as u32);
                    let mut current_offset: u8 = 0;
                    for i in 0..8 {
                        if freemask & (1 << i) == 0 {
                            continue;
                        }
                        let octree = octrees[octree_indexes[i]].as_mut().unwrap();
                        octree.1 = child_handle.offset(current_offset as u32);
                        let src_handle = octree.0.root;
                        let src = octree.0.arena.get(src_handle).clone();
                        let dst_handle = child_handle.offset(current_offset as u32);
                        let dst_ptr = arena.get_mut(dst_handle);
                        *dst_ptr = src;
                        current_offset += 1;
                    }
                    child_handle
                } else {
                    Handle::none()
                };
                Node {
                    _reserved: 0,
                    freemask,
                    _reserved2: 0,
                    children: child_handle,
                    data
                }
            } else {
                // Induction Step
                let mut nodes: [Node<T>; 8] = unsafe { std::mem::zeroed() };
                let mut data: [T; 8] = Default::default();
                let mut freemask: u8 = 0;
                let mut num_child: u8 = 0;
                for (i, corner) in Corner::all().enumerate() {
                    nodes[i] = generate(octrees, arena, bounds.half(corner));
                    let node: &Node<T> = &nodes[i];
                    data[i] = Voxel::avg(&node.data);
                    if node.freemask != 0 {
                        freemask |= (1 << i);
                        num_child += 1;
                    }
                }
                if num_child > 0 {
                    let children = arena.alloc(num_child as u32);
                    arena.changed_block(children, num_child as u32);
                    let mut current_index = 0;
                    for i in 0..8 {
                        let node: &Node<T> = &nodes[i];
                        if node.freemask == 0 {
                            continue;
                        }
                        *arena.get_mut(children.offset(current_index)) = node.clone();
                        // TODO: remove the clone with fixed-sized array into_iter
                        current_index += 1;
                    }
                    Node {
                        _reserved: 0,
                        freemask,
                        _reserved2: 0,
                        children,
                        data
                    }
                } else {
                    Node {
                        _reserved: 0,
                        freemask: 0,
                        _reserved2: 0,
                        children: Handle::none(),
                        data
                    }
                }

            }
        };

        let new_root = generate(&mut self.octrees, &mut self.arena, Bounds::with_width(GRID_SIZE));
        // TODO: free the old tree.
        *self.arena.get_mut(ROOT_HANDLE) = new_root;
        self.arena.changed(ROOT_HANDLE);
    }
    pub fn flush(&mut self) {
        if !self.block_allocator.can_flush() {
            return;
        }
        for octree in self.octrees.iter() {
            if let Some((octree, root)) = octree.as_ref() {
                let supertree_root = self.arena.get_mut(*root);
                let octree_root = octree.arena.get(octree.root);

                if supertree_root != octree_root {
                    *supertree_root = octree_root.clone();
                    self.arena.changed(*root);
                }
            }
        }
        let supertree_changelist = self.arena.flush();
        let mut iterator = self.octrees
            .iter_mut()
            .filter(|octree| octree.is_some())
            .map(|octree| {
                let (octree, root) = octree.as_mut().unwrap();
                octree
            })
            .flat_map(|octree| {
                octree.arena.flush().into_iter()
            })
            .chain(supertree_changelist.into_iter());
        unsafe {
            self.block_allocator.flush(&mut iterator);
        }
    }
}
