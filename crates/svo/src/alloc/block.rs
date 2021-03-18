use crate::{AllocError, BlockAllocator, MAX_BUFFER_SIZE};
use std::ops::Range;
use std::ptr::NonNull;
use std::collections::HashMap;
//use crate::alloc::{handle_alloc_error, AllocError, Allocator, Global, Layout, WriteCloneIntoRaw};
use std::alloc::{Allocator, Global, Layout};

pub struct SystemBlockAllocator<const SIZE: usize, A: Allocator = Global> {
    allocator: A,
}

impl<const SIZE: usize> BlockAllocator<SIZE>
for SystemBlockAllocator<SIZE>
{
    unsafe fn allocate_block(&mut self) -> Result<NonNull<[u8; SIZE]>, AllocError> {
        let layout = Layout::new::<[u8; SIZE]>();
        let mem = self.allocator.allocate(layout).map_err(|_| AllocError::OutOfHostMemory)?;
        let mem: NonNull<[u8; SIZE]> = unsafe { mem.cast() };
        Ok(mem)
    }

    unsafe fn deallocate_block(&mut self, block: NonNull<[u8; SIZE]>) {
        let layout = Layout::new::<[u8; SIZE]>();
        let ptr: NonNull<u8> = unsafe { block.cast() };
        self.allocator.deallocate(ptr, layout);
    }

    unsafe fn updated_block(&mut self, _block: NonNull<[u8; SIZE]>, _block_range: Range<u64>) {
    }

    unsafe fn flush(&mut self) {}
}
