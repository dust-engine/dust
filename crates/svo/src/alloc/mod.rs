mod arena;
mod system;
mod changeset;

pub use arena::CHUNK_DEGREE;
pub use arena::CHUNK_SIZE;
pub use arena::{ArenaAllocated, ArenaAllocator, ArenaBlockAllocator, Handle};
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
    unsafe fn flush(&mut self, ranges: &mut dyn Iterator<Item = (NonNull<[u8; SIZE]>, Range<u32>)>);
}
