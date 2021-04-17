mod arena;
mod changeset;
mod system;

pub use arena::{ArenaAllocated, ArenaAllocator, ArenaBlockAllocator, Handle};
pub use arena::{BLOCK_MASK, BLOCK_MASK_DEGREE, BLOCK_SIZE, NUM_SLOTS_IN_BLOCK};
use std::ops::Range;

pub use system::SystemBlockAllocator;

pub use changeset::ChangeSet;

#[derive(Debug)]
pub enum AllocError {
    OutOfHostMemory,
    OutOfDeviceMemory,
    MappingFailed,
    TooManyObjects,
}

#[derive(Clone)]
pub struct BlockAllocation(pub u64);
impl Drop for BlockAllocation {
    fn drop(&mut self) {
        panic!("BlockAllocation must be returned to the BlockAllocator!")
    }
}

pub trait BlockAllocator: Send + Sync {
    // Returns ptr, allocation, block index
    unsafe fn allocate_block(&self) -> Result<(*mut u8, BlockAllocation, u32), AllocError>;
    unsafe fn deallocate_block(&self, block: BlockAllocation);
    unsafe fn flush(&self, ranges: &mut dyn Iterator<Item = (BlockAllocation, Range<u32>)>);
    fn can_flush(&self) -> bool;
}
