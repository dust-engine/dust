use crate::alloc::Handle;
use std::ops::Range;
use std::collections::HashSet;
use std::mem::MaybeUninit;

#[derive(Clone, Debug)]
struct BlockChangeSet {
    range: Range<u32>,
}
impl BlockChangeSet {
    fn changed(&mut self, num: u32, len: u32) {
        self.range.start = self.range.start.min(num);
        self.range.end = self.range.end.max(num);
    }
    fn reset(&mut self, to: u32, len: u32) {
        self.range = to..(to + len);
    }
}

pub struct ChangeSet {
    chunks: Vec<BlockChangeSet>,
    changed_chunks: HashSet<u32>,
}

impl ChangeSet {
    pub fn new(len: usize) -> Self {
        ChangeSet {
            chunks: vec![BlockChangeSet { range: 0..0 }; len],
            changed_chunks: HashSet::new(),
        }
    }
    pub fn changed(&mut self, index: Handle) {
        self.changed_block(index, 1);
    }
    pub fn changed_block(&mut self, index: Handle, len: u32) {
        let chunk_num = index.get_chunk_num();
        let slot_num = index.get_slot_num();
        let chunk = &mut self.chunks[chunk_num as usize];
        if self.changed_chunks.contains(&chunk_num) {
            chunk.changed(slot_num, len);
        } else {
            chunk.reset(slot_num, len);
            self.changed_chunks.insert(chunk_num);
        }
    }
    pub fn reset(&mut self) {
        self.changed_chunks.clear();
    }
    pub fn add_chunk(&mut self) {
        self.chunks.push(BlockChangeSet {
            range: 0..0
        })
    }
}
