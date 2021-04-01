mod arena;
mod changeset;
mod system;

pub use arena::CHUNK_DEGREE;
pub use arena::CHUNK_SIZE;
pub use arena::{ArenaAllocated, ArenaAllocator, ArenaBlockAllocator, Handle};
use std::ops::Range;

pub use system::SystemBlockAllocator;

#[derive(Debug)]
pub enum AllocError {
    OutOfHostMemory,
    OutOfDeviceMemory,
    MappingFailed,
    TooManyObjects,
}

pub trait BlockAllocator: Send + Sync {
    unsafe fn allocate_block(&mut self) -> Result<*mut u8, AllocError>;
    unsafe fn deallocate_block(&mut self, block: *mut u8);
    unsafe fn flush(&mut self, ranges: &mut dyn Iterator<Item = (*mut u8, Range<u32>)>);
}
