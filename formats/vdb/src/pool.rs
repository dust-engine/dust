use std::{alloc::Layout, mem::size_of};

pub struct Pool {
    /// Size of one individual allocation
    size: usize,
    align: usize,
    /// Head of freelist
    head: u32,

    /// Top of free items.
    top: u32,
    /// Number of items to request when we run out of space.
    /// When running out of space, request (1 << chunk_size_log2) * size bytes.
    chunk_size_log2: usize,
    chunks: Vec<*mut u8>,
}

/// ```
/// use std::alloc::Layout;
/// use dust_vdb::Pool;
/// let item: u64 = 0;
/// // Create a pool of u64s with 2 items in each block.
/// let mut pool = Pool::new(Layout::for_value(&item), 1);
/// assert_eq!(pool.alloc(), 0);
/// assert_eq!(pool.alloc(), 1);
/// assert_eq!(pool.alloc(), 2);
/// assert_eq!(pool.alloc(), 3);
/// assert_eq!(pool.num_chunks(), 2);
///
/// pool.free(1);
/// pool.free(2);
/// assert_eq!(pool.alloc(), 2);
/// assert_eq!(pool.alloc(), 1);
/// assert_eq!(pool.alloc(), 4);
/// ```
impl Pool {
    pub fn new(layout: Layout, chunk_size_log2: usize) -> Self {
        Self {
            size: layout.pad_to_align().size(),
            align: layout.align(),
            head: u32::MAX,
            top: 0,
            chunk_size_log2,
            chunks: Vec::new(),
        }
    }
    pub fn alloc(&mut self) -> u32 {
        if self.head == u32::MAX {
            // allocate new
            let top = self.top;
            let chunk_index = top as usize >> self.chunk_size_log2;
            if chunk_index >= self.chunks.len() {
                // allocate new block
                unsafe {
                    let (layout, _) = Layout::from_size_align_unchecked(self.size, self.align)
                        .repeat(1 << self.chunk_size_log2)
                        .unwrap();
                    let block = std::alloc::alloc(layout);
                    println!("Allocated {:?}", block);
                    self.chunks.push(block);
                }
            }
            self.top += 1;
            top
        } else {
            // take from freelist
            unsafe {
                let item_location = self.get_mut(self.head);
                let next_available_location = *(item_location as *const u32);
                let head = self.head;
                self.head = next_available_location;
                return head;
            }
        }
    }
    pub fn free(&mut self, index: u32) {
        unsafe {
            // push to freelist
            let current_free_location = self.get_mut(index);
            *(current_free_location as *mut u32) = self.head;
            self.head = index;
        }
    }

    pub fn num_chunks(&self) -> usize {
        self.chunks.len()
    }

    #[inline]
    pub unsafe fn get(&self, ptr: u32) -> *const u8 {
        let chunk_index = (ptr as usize) >> self.chunk_size_log2;
        let item_index = (ptr as usize) & ((1 << self.chunk_size_log2) - 1);
        return self.chunks[chunk_index].add(item_index * self.size);
    }
    #[inline]
    pub unsafe fn get_mut(&mut self, ptr: u32) -> *mut u8 {
        let ptr = self.get(ptr);
        ptr as *mut u8
    }

    #[inline]
    pub unsafe fn get_item<T>(&self, ptr: u32) -> &T {
        debug_assert_eq!(Layout::new::<T>().pad_to_align(), Layout::from_size_align_unchecked(self.size, self.align));
        &*(self.get(ptr) as *const T)
    }
    #[inline]
    pub unsafe fn get_item_mut<T>(&mut self, ptr: u32) -> &mut T {
        debug_assert_eq!(Layout::new::<T>().pad_to_align(), Layout::from_size_align_unchecked(self.size, self.align));
        &mut *(self.get_mut(ptr) as *mut T)
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        unsafe {
            let (layout, _) = Layout::from_size_align_unchecked(self.size, self.align)
                .repeat(1 << self.chunk_size_log2)
                .unwrap();
            for chunk in self.chunks.iter() {
                let chunk = *chunk;
                std::alloc::dealloc(chunk, layout);
            }
        }
    }
}
