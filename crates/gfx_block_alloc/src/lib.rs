#![feature(nonnull_slice_from_raw_parts)]
mod discrete;
mod integrated;
mod utils;
use gfx_hal::prelude::*;

use gfx_hal as hal;
use std::ops::Range;
use std::ptr::NonNull;

const MAX_BUFFER_SIZE: u64 = 1 << 32;

// maxMemoryAllocationCount is typically 4096. We set the allocation block size to 16MB.
const ALLOCATION_BLOCK_SIZE: usize = 1 << 24;

#[derive(Debug)]
pub enum AllocError {
    OutOfHostMemory,
    OutOfDeviceMemory,
    MappingFailed,
    TooManyObjects,
}

pub trait BlockAllocator<const SIZE: usize> {
    type Block;
    unsafe fn allocate_block(&mut self) -> Result<Self::Block, AllocError>;
    unsafe fn deallocate_block(&mut self, block: Self::Block);
    unsafe fn updated_block(&mut self, block: &Self::Block, block_range: Range<u64>);
    unsafe fn flush(&mut self);
}
