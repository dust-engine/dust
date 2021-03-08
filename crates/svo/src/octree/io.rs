use crate::{Arena, Handle, Octree, Voxel};
use std::collections::VecDeque;
use std::io::{Read, Write, Seek, SeekFrom};
use std::mem::size_of;
use std::slice::{from_raw_parts, from_raw_parts_mut};

/// File structures:
/// root_data
/// NODE DATA
/// fences[]
/// fence_size
/// EOF
impl<T: Voxel> Octree<T> {
    pub fn write<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        // Writing some metadata
        unsafe {
            writer.write(from_raw_parts(
                &self.root_data as *const T as *const u8,
                size_of::<T>(),
            ))?;
        }
        // starting to DFS
        let mut queue: VecDeque<(Handle<T>, u8, u8)> = VecDeque::new(); // todo: optimize this with_capacity
        let mut current_index: u32 = 1; // The file address of the next available slot.
        let mut current_lod: u8 = 0;
        let mut fences: Vec<u32> = Vec::new();
        queue.push_back((self.root, 1, 0));
        while !queue.is_empty() {
            let (nodes, num_of_children, lod) = queue.pop_front().unwrap();
            let new_lod = lod + 1;
            if new_lod > current_lod {
                current_lod = new_lod;
                fences.push(current_index);
            }

            for i in 0..num_of_children {
                // For each children of this block
                let node = nodes.offset(i as u32);
                let node_ref = &self.arena[node];

                // Write the node into the file.
                unsafe {
                    // We're having three write() calls because we want to write something
                    // different as the children index.
                    writer.write(from_raw_parts::<u8>(&node_ref.freemask, size_of::<u8>()))?;

                    if node_ref.freemask != 0 {
                        let child_block_size = node_ref.freemask.count_ones();
                        // non leaf node.
                        // Add the children of the current node to the queue.
                        queue.push_back((node_ref.children, child_block_size as u8, new_lod));
                        // Translate the child index into the file space
                        writer.write(from_raw_parts::<u8>(
                            &current_index as *const u32 as *const u8,
                            size_of::<u32>(),
                        ))?;
                        current_index += child_block_size;
                    }

                    writer.write(from_raw_parts::<u8>(
                        &node_ref.data as *const T as *const u8,
                        size_of::<[T; 8]>(),
                    ))?;
                }
            }
        }
        unsafe {
            // Writing the fence
            let fence_size = fences.len() as u8;
            writer.write(from_raw_parts(
                fences.as_ptr() as *const u8,
                size_of::<u32>() * fences.len(),
            ))?;
            writer.write(from_raw_parts(
                &fence_size,
                size_of::<u8>(),
            ))?;
        }
        Ok(())
    }

    pub fn read<R: Read + Seek>(reader: &mut R, lod: u8) -> std::io::Result<Self> {
        let mut octree = Octree {
            arena: Arena::new(),
            root: Handle::new(0, 0),
            root_data: Default::default(),
        };

        let fences = unsafe {
            let original_pos = reader.stream_position()?;
            reader.seek(SeekFrom::End(-(size_of::<u8>() as i64)))?;

            let mut fence_size: u8 = 0;
            reader.read_exact(from_raw_parts_mut(
                &mut fence_size,
                size_of::<u8>(),
            ))?;

            reader.seek(SeekFrom::End(-((size_of::<u8>() + size_of::<u32>() * fence_size as usize) as i64)))?;
            let mut vec: Vec<u32> = vec![0; fence_size as usize];
            reader.read_exact(from_raw_parts_mut(
                vec.as_mut_ptr() as *mut u8,
                size_of::<u32>() * vec.len(),
            ))?;
            reader.seek(SeekFrom::Start(0))?;
            vec
        };

        unsafe {
            // Read the root data
            reader.read_exact(from_raw_parts_mut(
                &mut octree.root_data as *mut T as *mut u8,
                size_of::<T>(),
            ))?;
        }

        // Mapping from file-space indices to (Parent, BlockSize)
        let mut block_size_map: VecDeque<(Handle<T>, u8)> = VecDeque::new(); // todo: optimize with_capacity
        // let mut block_size_map: AHashMap<u32, (Handle<T>, u8)> = AHashMap::new();
        // The root node is always the first one in the file, and the block size of the root node
        // is always one.
        block_size_map.push_back((Handle::none(), 1));
        let mut slots_loaded: u32 = 0;
        let total_slots = if lod as usize >= fences.len() {
            u32::MAX
        } else {
            fences[lod as usize]
        };
        let total_nonleaf_slots = if (lod-1) as usize >= fences.len() {
            u32::MAX
        } else {
            fences[(lod-1) as usize]
        };
        while !block_size_map.is_empty() {
            let (parent_handle, block_size) = block_size_map.pop_front().unwrap();

            let block = octree.arena.alloc(block_size as u32);
            if !parent_handle.is_none() {
                // Has a parent. Set the parent's child index to convert it back into memory space
                let parent_ref = &mut octree.arena[parent_handle];
                parent_ref.children = block;
            }

            for i in 0..block_size {
                let node = block.offset(i as u32);
                let node_ref = &mut octree.arena[node];
                node_ref.block_size = block_size;
                unsafe {
                    // Read the entire thing into the newly allocated node
                    reader.read_exact(from_raw_parts_mut::<u8>(
                        &mut node_ref.freemask,
                        size_of::<u8>(),
                    ))?;
                    if slots_loaded < total_nonleaf_slots {
                        // non-leaf slot
                        if node_ref.freemask != 0 {
                            // has children
                            reader.read_exact(from_raw_parts_mut::<u8>(
                                &mut node_ref.children as *mut Handle<T> as *mut u8,
                                size_of::<Handle<T>>(),
                            ))?;
                            block_size_map.push_back((node, node_ref.freemask.count_ones() as u8));
                        }
                    } else {
                        // leaf slot
                        if node_ref.freemask != 0 {
                            // has children
                            reader.seek(SeekFrom::Current(size_of::<Handle<T>>() as i64))?;
                        }
                        node_ref.freemask = 0; // Force to be a leaf
                    }

                    reader.read_exact(from_raw_parts_mut::<u8>(
                        &mut node_ref.data as *mut T as *mut u8,
                        size_of::<[T; 8]>(),
                    ))?;
                }
            }
            slots_loaded += block_size as u32;
        }
        Ok(octree)
    }
}
