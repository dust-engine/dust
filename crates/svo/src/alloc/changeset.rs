use crate::alloc::Handle;
use std::collections::HashMap;

use std::ops::{Range, DerefMut};
use std::cell::Cell;
use crossbeam::atomic::AtomicCell;
use std::sync::RwLock;

#[derive(Clone, Debug)]
struct BlockChangeSet {
    range: Range<u32>,
}
impl BlockChangeSet {
    fn changed(&mut self, num: u32, len: u32) {
        self.range.start = self.range.start.min(num);
        self.range.end = self.range.end.max(num + len);
    }
    fn new(to: u32, len: u32) -> Self {
        BlockChangeSet {
            range: to..(to + len),
        }
    }
}

pub struct ChangeSet {
    changed_chunks: RwLock<HashMap<u32, BlockChangeSet>>,
}

impl ChangeSet {
    pub fn new(_len: usize) -> Self {
        ChangeSet {
            changed_chunks: RwLock::new(HashMap::new()),
        }
    }
    pub fn changed(&mut self, index: Handle) {
        self.changed_block(index, 1);
    }
    pub fn changed_block(&mut self, index: Handle, len: u32) {
        let chunk_num = index.get_chunk_num();
        let slot_num = index.get_slot_num();

        let mut changed_chunks = self.changed_chunks.write().unwrap();
        if let Some(chunk) = changed_chunks.get_mut(&chunk_num) {
            chunk.changed(slot_num, len);
        } else {
            changed_chunks.insert(chunk_num, BlockChangeSet::new(slot_num, len));
        }
    }

    // returns: iterator of (chunk_index, range of slots)
    // Safety: can't be called from multiple threads. However it's ok to call drain when the block
    // was being mutated by another thread using changed_block.?????
    pub fn drain<'a>(&self) -> Vec<(u32, Range<u32>)> {
        let mut old_hashmap = HashMap::new();
        let mut guard = self.changed_chunks.write().unwrap();
        std::mem::swap(&mut old_hashmap, guard.deref_mut());
        drop(guard);
        old_hashmap.drain()
            .map(|(key, val)| (key, val.range))
            .collect()
    }
}
