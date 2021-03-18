use std::ops::Range;
use std::ptr::NonNull;
//use crate::alloc::{handle_alloc_error, AllocError, Allocator, Global, Layout, WriteCloneIntoRaw};
use super::AllocError;
use super::BlockAllocator;
use std::alloc::{Allocator, Global, Layout};

pub struct SystemBlockAllocator<const SIZE: usize, A: Allocator = Global> {
    allocator: A,
}

impl<const SIZE: usize> SystemBlockAllocator<SIZE> {
    pub fn new() -> SystemBlockAllocator<SIZE, Global> {
        SystemBlockAllocator {
            allocator: Global
        }
    }
}

impl<const SIZE: usize> BlockAllocator<SIZE> for SystemBlockAllocator<SIZE> {
    unsafe fn allocate_block(&mut self) -> Result<NonNull<[u8; SIZE]>, AllocError> {
        let layout = Layout::new::<[u8; SIZE]>();
        let mem = self
            .allocator
            .allocate(layout)
            .map_err(|_| AllocError::OutOfHostMemory)?;
        Ok(mem.cast())
    }

    unsafe fn deallocate_block(&mut self, block: NonNull<[u8; SIZE]>) {
        let layout = Layout::new::<[u8; SIZE]>();
        self.allocator.deallocate(block.cast(), layout);
    }

    unsafe fn updated_block(&mut self, _block: NonNull<[u8; SIZE]>, _block_range: Range<u64>) {}

    unsafe fn flush(&mut self) {}
}
