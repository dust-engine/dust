use crate::alloc::Handle;
use std::collections::HashMap;

use std::ops::Range;

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
    changed_chunks: HashMap<u32, BlockChangeSet>,
}

impl ChangeSet {
    pub fn new(_len: usize) -> Self {
        ChangeSet {
            changed_chunks: HashMap::new(),
        }
    }
    pub fn changed(&mut self, index: Handle) {
        self.changed_block(index, 1);
    }
    pub fn changed_block(&mut self, index: Handle, len: u32) {
        let chunk_num = index.get_chunk_num();
        let slot_num = index.get_slot_num();
        if let Some(chunk) = self.changed_chunks.get_mut(&chunk_num) {
            chunk.changed(slot_num, len);
        } else {
            self.changed_chunks
                .insert(chunk_num, BlockChangeSet::new(slot_num, len));
        }
    }

    // returns: iterator of (chunk_index, range of slots)
    pub fn drain<'a>(&'a mut self) -> impl Iterator<Item = (u32, Range<u32>)> + 'a {
        self.changed_chunks
            .drain()
            .map(move |(i, changes)| (i, changes.range.clone()))
    }
    pub fn len(&self) -> usize {
        self.changed_chunks.len()
    }
}
