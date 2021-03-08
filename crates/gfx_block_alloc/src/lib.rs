#![feature(nonnull_slice_from_raw_parts)]
mod discrete;
mod utils;
use gfx_hal::prelude::*;

use gfx_hal as hal;
use gfx_hal::device::WaitFor::All;
use std::ptr::NonNull;

const MAX_BUFFER_SIZE: u64 = 1 << 32;

// maxMemoryAllocationCount is typically 4096. We set the allocation block size to 16MB.
const ALLOCATION_BLOCK_SIZE: usize = 1 << 24;

pub enum AllocError {
    OutOfHostMemory,
    OutOfDeviceMemory,
    MappingFailed,
    TooManyObjects,
}

pub trait BlockAllocator<B: hal::Backend, const SIZE: usize> {
    type Block;
    unsafe fn allocate_block(&self) -> Result<Self::Block, AllocError>;
    unsafe fn deallocate_block(&self, block: Self::Block);
}
