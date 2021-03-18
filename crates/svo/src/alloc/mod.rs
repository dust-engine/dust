mod arena;
mod system;

pub use arena::CHUNK_DEGREE;
pub use arena::CHUNK_SIZE;
pub use arena::{ArenaAllocated, ArenaAllocator, Handle};
use std::ops::Range;
use std::ptr::NonNull;
pub use system::SystemBlockAllocator;

#[derive(Debug)]
pub enum AllocError {
    OutOfHostMemory,
    OutOfDeviceMemory,
    MappingFailed,
    TooManyObjects,
}

pub trait BlockAllocator<const SIZE: usize> {
    unsafe fn allocate_block(&mut self) -> Result<NonNull<[u8; SIZE]>, AllocError>;
    unsafe fn deallocate_block(&mut self, block: NonNull<[u8; SIZE]>);
    unsafe fn updated_block(&mut self, block: NonNull<[u8; SIZE]>, block_range: Range<u64>);
    unsafe fn flush(&mut self);
}
