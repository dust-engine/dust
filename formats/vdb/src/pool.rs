use std::alloc::Layout;

pub struct Pool {
    /// Size of one individual allocation
    layout: Layout,
    /// Head of freelist
    head: u32,

    /// Top of free items.
    top: u32,
    /// Number of items to request when we run out of space.
    /// When running out of space, request (1 << chunk_size_log2) * size bytes.
    chunk_size_log2: usize,
    chunks: Vec<*mut u8>,
}

/// A memory pool for objects of the same layout.
/// ```
/// use std::alloc::Layout;
/// use dust_vdb::Pool;
/// let item: u64 = 0;
/// // Create a pool of u64s with 2 items in each block.
/// unsafe {
///   let mut pool = Pool::new(Layout::for_value(&item), 1);
///   assert_eq!(pool.alloc::<u64>(), 0);
///   assert_eq!(pool.alloc::<u64>(), 1);
///   assert_eq!(pool.alloc::<u64>(), 2);
///   assert_eq!(pool.alloc::<u64>(), 3);
///   assert_eq!(pool.num_chunks(), 2);
///
///   pool.free(1);
///   pool.free(2);
///   assert_eq!(pool.alloc::<u64>(), 2);
///   assert_eq!(pool.alloc::<u64>(), 1);
///   assert_eq!(pool.alloc::<u64>(), 4);
/// }
/// ```
impl Pool {
    pub fn new(layout: Layout, chunk_size_log2: usize) -> Self {
        Self {
            layout: layout.pad_to_align(),
            head: u32::MAX,
            top: 0,
            chunk_size_log2,
            chunks: Vec::new(),
        }
    }
    pub unsafe fn alloc<T: Default>(&mut self) -> u32 {
        debug_assert_eq!(Layout::new::<T>(), self.layout);
        let ptr = self.alloc_uninitialized();
        let item = self.get_item_mut::<T>(ptr);
        *item = T::default();
        ptr
    }
    pub unsafe fn alloc_uninitialized(&mut self) -> u32 {
        if self.head == u32::MAX {
            // allocate new
            let top = self.top;
            let chunk_index = top as usize >> self.chunk_size_log2;
            if chunk_index >= self.chunks.len() {
                // allocate new block
                let (layout, _) = self.layout.repeat(1 << self.chunk_size_log2).unwrap();
                let block = std::alloc::alloc(layout);
                self.chunks.push(block);
            }
            self.top += 1;
            top
        } else {
            // take from freelist
            let item_location = self.get_mut(self.head);
            let next_available_location = *(item_location as *const u32);
            let head = self.head;
            self.head = next_available_location;
            return head;
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
        return self.chunks.get_unchecked(chunk_index).add(item_index * self.layout.size());
    }
    #[inline]
    pub unsafe fn get_mut(&mut self, ptr: u32) -> *mut u8 {
        let ptr = self.get(ptr);
        ptr as *mut u8
    }

    #[inline]
    pub unsafe fn get_item<T>(&self, ptr: u32) -> &T {
        debug_assert_eq!(Layout::new::<T>().pad_to_align(), self.layout);
        &*(self.get(ptr) as *const T)
    }
    #[inline]
    pub unsafe fn get_item_mut<T>(&mut self, ptr: u32) -> &mut T {
        debug_assert_eq!(Layout::new::<T>().pad_to_align(), self.layout);
        &mut *(self.get_mut(ptr) as *mut T)
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        unsafe {
            let (layout, _) = self.layout.repeat(1 << self.chunk_size_log2).unwrap();
            for chunk in self.chunks.iter() {
                let chunk = *chunk;
                std::alloc::dealloc(chunk, layout);
            }
        }
    }
}
