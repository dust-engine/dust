use std::ops::Range;
use std::ptr::NonNull;
//use crate::alloc::{handle_alloc_error, AllocError, Allocator, Global, Layout, WriteCloneIntoRaw};
use super::AllocError;
use super::BlockAllocator;
use std::alloc::{Allocator, Global, Layout};

pub struct SystemBlockAllocator<A: Allocator = Global> {
    allocator: A,
    block_size: u32,
}

impl SystemBlockAllocator {
    pub fn new(block_size: u32) -> SystemBlockAllocator<Global> {
        SystemBlockAllocator {
            allocator: Global,
            block_size,
        }
    }
}

impl BlockAllocator for SystemBlockAllocator {
    unsafe fn allocate_block(&mut self) -> Result<*mut u8, AllocError> {
        let mem = self
            .allocator
            .allocate(Layout::from_size_align_unchecked(
                self.block_size as usize,
                1,
            ))
            .map_err(|_| AllocError::OutOfHostMemory)?;
        Ok(mem.as_mut_ptr())
    }

    unsafe fn deallocate_block(&mut self, block: *mut u8) {
        let layout = Layout::new::<u8>()
            .repeat(self.block_size as usize)
            .unwrap();
        self.allocator.deallocate(
            NonNull::new(block).unwrap(),
            Layout::from_size_align_unchecked(self.block_size as usize, 1),
        );
    }

    unsafe fn flush(&mut self, ranges: &mut dyn Iterator<Item = (*mut u8, Range<u32>)>) {}
}
