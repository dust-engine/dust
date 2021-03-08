use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ops::{Index, IndexMut};

use crate::Corner;

const BLOCK_SIZE: u32 = 13;
type Block<T> = [Slot<T>; 1 << BLOCK_SIZE];

#[derive(Copy, Clone)]
pub struct ArenaHandle<T: Copy> {
    _marker: PhantomData<T>,
    pub(crate) index: u32,
}

impl<T: Copy> ArenaHandle<T> {
    pub const fn none() -> Self {
        ArenaHandle {
            _marker: PhantomData,
            index: std::u32::MAX,
        }
    }
    #[inline]
    pub fn is_none(&self) -> bool {
        self.index == std::u32::MAX
    }
    pub(crate) fn new(block_num: u32, item_num: u32) -> Self {
        ArenaHandle {
            _marker: PhantomData,
            index: (block_num << BLOCK_SIZE) | item_num,
        }
    }
    pub(crate) fn offset(&self, n: u32) -> Self {
        let (block_num, item_num): (u32, u32) = self.into();
        ArenaHandle::new(block_num, item_num + n)
    }
}

impl<T: Copy> From<ArenaHandle<T>> for (u32, u32) {
    fn from(handle: ArenaHandle<T>) -> Self {
        let item_num = handle.index & ((1 << BLOCK_SIZE) - 1);
        let block_num = handle.index >> BLOCK_SIZE;
        (block_num, item_num)
    }
}

impl<T: Copy> From<&ArenaHandle<T>> for (u32, u32) {
    fn from(handle: &ArenaHandle<T>) -> Self {
        (*handle).into()
    }
}

impl<T: Copy> From<ArenaHandle<T>> for (usize, usize) {
    fn from(handle: ArenaHandle<T>) -> Self {
        let (block_num, item_num): (u32, u32) = handle.into();
        (block_num as usize, item_num as usize)
    }
}

impl<T: Copy> std::fmt::Debug for ArenaHandle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        let (block_num, item_num): (u32, u32) = self.into();
        f.write_fmt(format_args!("ArenaHandle({:?}, {:?})", block_num, item_num))
    }
}

impl<T: Copy> std::cmp::PartialEq for ArenaHandle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl<T: Copy> Eq for ArenaHandle<T> {}

#[repr(C)]
struct FreeSlot<T: Copy> {
    pub(crate) block_size: u8, // This value is 0 for free blocks
    freemask: u8,              // 0 means no children, 1 means has children
    _reserved: u16,
    next: ArenaHandle<T>, // 32 bits
}

#[repr(C)]
pub struct NodeSlot<T: Copy> {
    pub(crate) block_size: u8, // This value is always OCCUPIED_FLAG for occupied nodes.
    pub freemask: u8,          // 0 means no children, 1 means has children
    _reserved2: u16,           // Alignment
    pub children: ArenaHandle<T>, // 32 bits
    pub data: [T; 8],
}

impl<T: Copy> NodeSlot<T> {
    pub fn child(&self, corner: Corner) -> ArenaHandle<T> {
        /// Given a mask an a location, returns n where the given '1' on the location
        /// is the nth '1' counting from the Least Significant Bit.
        fn mask_location_nth_one(mask: u8, location: u8) -> u8 {
            (mask & ((1 << location) - 1)).count_ones() as u8
        }
        self.children
            .offset(mask_location_nth_one(self.freemask, corner as u8) as u32)
    }
}

union Slot<T: Copy> {
    occupied: NodeSlot<T>,
    free: FreeSlot<T>,
}

pub struct Arena<T: Copy> {
    data: Vec<Box<Block<T>>>,
    freelist_heads: [ArenaHandle<T>; 8],
    newspace_top: ArenaHandle<T>, // new space to be allocated
    pub(crate) size: u32,         // number of allocated slots
    pub(crate) num_blocks: u32,   // number of allocated blocks
    pub(crate) capacity: u32,     // number of available slots
}

impl<T: Copy> Arena<T> {
    pub fn new() -> Arena<T> {
        Arena {
            data: vec![],
            freelist_heads: [
                ArenaHandle::none(),
                ArenaHandle::none(),
                ArenaHandle::none(),
                ArenaHandle::none(),
                ArenaHandle::none(),
                ArenaHandle::none(),
                ArenaHandle::none(),
                ArenaHandle::none(),
            ],
            // Space pointed by this is guaranteed to have free space > 8
            newspace_top: ArenaHandle::none(),
            size: 0,
            num_blocks: 0,
            capacity: 0,
        }
    }
    pub fn alloc(&mut self, len: u32) -> ArenaHandle<T> {
        assert!(0 < len && len <= 8, "Only supports block size between 1-8!");
        self.size += len;
        self.num_blocks += 1;
        let sized_head = self.freelist_heads[len as usize - 1];
        let handle = if sized_head.is_none() {
            // Put it in a new space
            if self.newspace_top.is_none() {
                let block_num = self.alloc_block();
                let handle = ArenaHandle::new(block_num, 0);
                self.newspace_top = ArenaHandle::new(block_num, len);
                handle
            } else {
                let handle = self.newspace_top;
                let (block_num, item_num): (u32, u32) = handle.into();
                let max_size = 1 << BLOCK_SIZE;
                let remaining_space = max_size - item_num - len;
                if remaining_space > 8 {
                    // There's still space
                    self.newspace_top = ArenaHandle::new(block_num, item_num + len);
                } else {
                    // Need to request new space
                    if remaining_space > 0 {
                        self.freelist_push(
                            remaining_space as u8 - 1,
                            ArenaHandle::new(block_num, item_num + len),
                        );
                    }
                    self.newspace_top = ArenaHandle::none();
                }
                handle
            }
        } else {
            self.freelist_heads[len as usize - 1] = unsafe { self.get_slot(sized_head).free.next };
            sized_head
        };

        let (block_num, item_num): (usize, usize) = handle.into();
        for i in item_num..(item_num + len as usize) {
            unsafe {
                // The block should be free
                // Using debug asserts because in release mode we don't initialize block_size field
                debug_assert_eq!(self.data[block_num][i].free.block_size, 0);
                self.data[block_num][i].free.block_size = len as u8;
                self.data[block_num][i].free.freemask = 0;
            }
        }
        handle
    }
    fn alloc_block(&mut self) -> u32 {
        let block_index = self.data.len() as u32;
        let block = Box::new(unsafe {
            MaybeUninit::<[Slot<T>; (1 << BLOCK_SIZE)]>::zeroed().assume_init()
        });
        self.data.push(block);
        self.capacity += 1 << BLOCK_SIZE;
        block_index
    }
    fn freelist_push(&mut self, n: u8, handle: ArenaHandle<T>) {
        let n = n as usize;
        self.get_slot_mut(handle).free.next = self.freelist_heads[n];
        self.freelist_heads[n] = handle;
    }
    fn get_slot(&self, handle: ArenaHandle<T>) -> &Slot<T> {
        let (block_num, item_num): (usize, usize) = handle.into();
        &self.data[block_num][item_num]
    }
    fn get_slot_mut(&mut self, handle: ArenaHandle<T>) -> &mut Slot<T> {
        let (block_num, item_num): (usize, usize) = handle.into();
        &mut self.data[block_num][item_num]
    }
    pub fn free(&mut self, handle: ArenaHandle<T>) {
        let block_size = unsafe { self.get_slot(handle).free.block_size };
        debug_assert!(block_size > 0, "Double free detected");
        let (block_num, item_num): (usize, usize) = handle.into();
        for i in item_num..(item_num + block_size as usize) {
            let block = &mut self.data[block_num][i];
            unsafe {
                // Detect double free
                debug_assert_eq!(
                    block.free.block_size, block_size,
                    "Overlapping handle detected"
                );
                block.occupied = std::mem::zeroed();
            }
        }
        self.freelist_push(block_size - 1, handle);
        self.size -= block_size as u32;
        self.num_blocks -= 1;
    }
    pub fn realloc(&mut self, node: ArenaHandle<T>, new_mask: u8) {
        let node_ref = &mut self[node];
        let old_mask = node_ref.freemask;
        let old_child_handle = node_ref.children;

        if old_mask == 0 && new_mask == 0 {
            return;
        }
        let new_num_items = new_mask.count_ones();
        if old_mask == 0 {
            let new_child_handle = self.alloc(new_num_items);
            let node_ref = &mut self[node];
            node_ref.freemask = new_mask;
            node_ref.children = new_child_handle;
            return;
        }
        if new_mask == 0 {
            let node_ref = &mut self[node];
            node_ref.freemask = 0;
            self.free(old_child_handle);
            return;
        }
        let new_child_handle = self.alloc(new_num_items);

        let mut new_slot_num: u8 = 0;
        let mut old_slot_num: u8 = 0;
        for i in 0..8 {
            let old_have_children_at_i = old_mask & (1 << i) != 0;
            let new_have_children_at_i = new_mask & (1 << i) != 0;
            if old_have_children_at_i && new_have_children_at_i {
                let (old_block_num, old_item_num): (usize, usize) = old_child_handle.into();
                let (new_block_num, new_item_num): (usize, usize) = new_child_handle.into();
                let new_item: *mut Slot<T> = &mut self.data[new_block_num]
                    [new_item_num + new_slot_num as usize]
                    as *mut Slot<T>;
                let old_item: *const Slot<T> = &self.data[old_block_num]
                    [old_item_num + old_slot_num as usize]
                    as *const Slot<T>;
                unsafe {
                    std::ptr::copy(old_item, new_item, 1);
                    let new_item = &mut *new_item;
                    new_item.occupied.block_size = new_num_items as u8;
                }
            }
            if old_have_children_at_i {
                old_slot_num += 1;
            }
            if new_have_children_at_i {
                new_slot_num += 1;
            }
        }

        let node_ref = &mut self[node];
        node_ref.freemask = new_mask;
        node_ref.children = new_child_handle;
        self.free(old_child_handle);
    }
    pub fn total_data_size(&self) -> usize {
        self.capacity as usize * std::mem::size_of::<Slot<T>>()
    }
    pub fn copy_data_into_slice(&self, slice: &mut [u8]) {
        for (i, block) in self.data.iter().enumerate() {
            let block_size = std::mem::size_of::<Block<T>>();
            let start = i * block_size;
            let end = start + block_size;
            let ptr = block.as_ptr(); // pointer to 1<<BLOCK_SIZE Slot<T>
            let ptr = ptr as *const u8;
            let data = unsafe { std::slice::from_raw_parts(ptr, block_size) };
            slice[start..end].copy_from_slice(data);
        }
    }
}

impl<T: Copy> Index<ArenaHandle<T>> for Arena<T> {
    type Output = NodeSlot<T>;
    fn index(&self, index: ArenaHandle<T>) -> &Self::Output {
        let (block_num, item_num): (usize, usize) = index.into();
        unsafe { &self.data[block_num][item_num].occupied }
    }
}

impl<T: Copy> IndexMut<ArenaHandle<T>> for Arena<T> {
    fn index_mut(&mut self, index: ArenaHandle<T>) -> &mut Self::Output {
        let (block_num, item_num): (usize, usize) = index.into();
        unsafe { &mut self.data[block_num][item_num].occupied }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;
    #[test]
    fn test_size() {
        assert_eq!(size_of::<Slot<u32>>(), 40);
        assert_eq!(size_of::<Slot<u32>>(), size_of::<Slot<u32>>());

        let mut slot: Slot<u32> = unsafe { MaybeUninit::uninit().assume_init() };
        unsafe {
            assert_eq!(
                &slot.free.block_size as *const u8,
                &slot.occupied.block_size as *const u8
            );
            let magic: u8 = 0b11100010;
            slot.occupied.block_size = magic;
            assert_eq!(slot.free.block_size, magic);
        }

        // So that one block maps to three GPU memory pages, 3 * 64K
        let block_size = std::mem::size_of::<Block<u16>>();
        let gpu_page_size = 64 * 1024;
        assert_eq!(block_size, 3 * gpu_page_size);
    }

    #[test]
    fn test_alloc() {
        {
            let mut arena: Arena<u32> = Arena::new();
            for i in 0..((1 << BLOCK_SIZE) - 8) {
                let handle = arena.alloc(1);
                let (block, item): (u32, u32) = handle.into();
                assert_eq!(item, i);
                assert_eq!(block, 0);
            }
            assert_eq!(arena.capacity, 1 << BLOCK_SIZE);
            for i in 0..10 {
                let handle = arena.alloc(1);
                let (block, item): (u32, u32) = handle.into();
                assert_eq!(item, i);
                assert_eq!(block, 1);
            }
            assert_eq!(arena.capacity, (1 << BLOCK_SIZE) * 2);
            assert_eq!(
                arena.freelist_heads[7],
                ArenaHandle::new(0, (1 << BLOCK_SIZE) - 8)
            );
            let handle = arena.alloc(5);
            let (block, item): (u32, u32) = handle.into();
            assert_eq!(item, 10);
            assert_eq!(block, 1);
            let handle = arena.alloc(8);
            let (block, item): (u32, u32) = handle.into();
            assert_eq!(item, (1 << BLOCK_SIZE) - 8);
            assert_eq!(block, 0);
        }
    }

    #[test]
    fn test_free() {
        let mut arena: Arena<u32> = Arena::new();
        let handles: Vec<ArenaHandle<u32>> = (0..8).map(|_| arena.alloc(4)).collect();
        for handle in handles.iter().rev() {
            arena.free(*handle);
        }
        assert_eq!(arena.alloc(1), ArenaHandle::new(0, 8 * 4));
        for handle in handles.iter() {
            let new_handle = arena.alloc(4);
            assert_eq!(*handle, new_handle);
        }
    }

    #[test]
    #[should_panic(expected = "Double free detected")]
    fn test_doublefree() {
        let mut arena: Arena<u32> = Arena::new();
        let handle = arena.alloc(3);
        arena.free(handle);
        arena.free(handle);
    }

    #[test]
    #[should_panic(expected = "Overlapping handle detected")]
    fn test_invalid_overlapping_handle() {
        let mut arena: Arena<u32> = Arena::new();
        let handle = arena.alloc(8);
        let (block_num, item_num): (u32, u32) = handle.into();
        let handle = ArenaHandle::new(block_num, item_num + 4);
        arena.free(handle);
    }

    #[test]
    fn test_realloc() {
        let mut arena: Arena<u32> = Arena::new();
        let node = arena.alloc(1);

        arena.realloc(node, 0b00001111);
        let node_ref = &mut arena[node];

        let childrens = [
            node_ref.child(Corner::RearLeftBottom),
            node_ref.child(Corner::FrontLeftBottom),
            node_ref.child(Corner::RearLeftTop),
            node_ref.child(Corner::FrontLeftTop),
        ];
        arena[childrens[0]].data[0] = 33;
        arena[childrens[1]].data[0] = 34;
        arena[childrens[2]].data[0] = 35;
        arena[childrens[3]].data[0] = 36;

        arena.realloc(node, 0b00011111);
        let node_ref = &mut arena[node];
        let childrens = [
            node_ref.child(Corner::RearLeftBottom),
            node_ref.child(Corner::FrontLeftBottom),
            node_ref.child(Corner::RearLeftTop),
            node_ref.child(Corner::FrontLeftTop),
        ];
        assert_eq!(arena[childrens[0]].data[0], 33);
        assert_eq!(arena[childrens[1]].data[0], 34);
        assert_eq!(arena[childrens[2]].data[0], 35);
        assert_eq!(arena[childrens[3]].data[0], 36);

        arena.realloc(node, 0b00001110);
        let node_ref = &mut arena[node];
        let childrens = [
            node_ref.child(Corner::FrontLeftBottom),
            node_ref.child(Corner::RearLeftTop),
            node_ref.child(Corner::FrontLeftTop),
        ];
        assert_eq!(arena[childrens[0]].data[0], 34);
        assert_eq!(arena[childrens[1]].data[0], 35);
        assert_eq!(arena[childrens[2]].data[0], 36);
    }
}
