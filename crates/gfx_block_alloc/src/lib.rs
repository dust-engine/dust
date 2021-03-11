#![feature(nonnull_slice_from_raw_parts)]
mod discrete;
mod integrated;
mod utils;

use std::ops::Range;
use std::ptr::NonNull;

pub use discrete::DiscreteBlock;
pub use discrete::DiscreteBlockAllocator;
pub use integrated::IntegratedBlock;
pub use integrated::IntegratedBlockAllocator;

const MAX_BUFFER_SIZE: u64 = 1 << 32;

#[derive(Debug)]
pub enum AllocError {
    OutOfHostMemory,
    OutOfDeviceMemory,
    MappingFailed,
    TooManyObjects,
}

pub trait AllocatorBlock<const SIZE: usize> {
    fn ptr(&self) -> NonNull<[u8; SIZE]>;
}

/// This is responsible for
pub trait BlockAllocator<const SIZE: usize> {
    type Block: AllocatorBlock<SIZE>;
    unsafe fn allocate_block(&mut self) -> Result<Self::Block, AllocError>;
    unsafe fn deallocate_block(&mut self, block: Self::Block);
    unsafe fn updated_block(&mut self, block: &Self::Block, block_range: Range<u64>);
    unsafe fn flush(&mut self);
}
