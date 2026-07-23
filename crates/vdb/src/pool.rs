use std::{alloc::Layout, any::Any};

pub trait PoolStorage: Send + Sync + Any {
    fn resize(&mut self, size: usize) -> *mut u8;
    fn device_address(&self) -> u64;
}

pub struct DefaultPoolStorage {
    size: usize,
    align: usize,
    buffer: *mut u8,
}
unsafe impl Send for DefaultPoolStorage {}
unsafe impl Sync for DefaultPoolStorage {}
impl PoolStorage for DefaultPoolStorage {
    fn device_address(&self) -> u64 {
        0
    }
    fn resize(&mut self, size: usize) -> *mut u8 {
        unsafe {
            let ptr = std::alloc::alloc(Layout::from_size_align_unchecked(size, self.align));
            if !self.buffer.is_null() {
                std::ptr::copy_nonoverlapping(self.buffer, ptr, self.size.min(size));
                std::alloc::dealloc(
                    self.buffer,
                    Layout::from_size_align_unchecked(self.size, self.align),
                );
            }
            self.size = size;
            self.buffer = ptr;
            ptr
        }
    }
}

pub struct Pool {
    /// Size of one individual allocation
    layout: Layout,
    /// Head of freelist
    head: u32,

    /// Top of free items.
    top: u32,

    /// Number of allocated items
    count: u32,

    /// Capacity of the underlying storage
    capacity: u32,

    /// Reference into the underlying storage buffer
    ptr: *mut u8,

    storage: Box<dyn PoolStorage>,
}

unsafe impl Send for Pool {}
unsafe impl Sync for Pool {}

/// A memory pool for objects of the same layout.
/// ```
/// use std::alloc::Layout;
/// use dust_vdb::pool::Pool;
/// let item: u64 = 0;
/// // Create a pool of u64s.
/// unsafe {
///   let mut pool = Pool::new(Layout::for_value(&item));
///   assert_eq!(pool.alloc::<u64>(), 0);
///   assert_eq!(pool.alloc::<u64>(), 1);
///   assert_eq!(pool.alloc::<u64>(), 2);
///   assert_eq!(pool.alloc::<u64>(), 3);
///
///   // Freed entries are recycled in LIFO order.
///   pool.free(1);
///   pool.free(2);
///   assert_eq!(pool.alloc::<u64>(), 2);
///   assert_eq!(pool.alloc::<u64>(), 1);
///   assert_eq!(pool.alloc::<u64>(), 4);
/// }
/// ```
impl Pool {
    pub fn new(layout: Layout) -> Self {
        Self {
            layout: layout.pad_to_align(),
            head: u32::MAX,
            top: 0,
            count: 0,
            capacity: 0,
            ptr: std::ptr::null_mut(),
            storage: Box::new(DefaultPoolStorage {
                size: 0,
                align: layout.align(),
                buffer: std::ptr::null_mut(),
            }),
        }
    }
    pub fn new_with_storage(layout: Layout, storage: Box<dyn PoolStorage>) -> Self {
        Self {
            layout: layout.pad_to_align(),
            head: u32::MAX,
            top: 0,
            count: 0,
            capacity: 0,
            ptr: std::ptr::null_mut(),
            storage,
        }
    }
    pub fn count(&self) -> u32 {
        self.count
    }
    pub fn storage(&self) -> &dyn PoolStorage {
        &*self.storage
    }

    pub fn storage_mut(&mut self) -> &mut dyn PoolStorage {
        &mut *self.storage
    }
    pub unsafe fn alloc<T: Default>(&mut self) -> u32 {
        unsafe {
            debug_assert_eq!(Layout::new::<T>(), self.layout);
            let ptr = self.alloc_uninitialized();
            let item = self.get_item_mut::<T>(ptr);
            *item = T::default();
            ptr
        }
    }
    pub unsafe fn alloc_uninitialized(&mut self) -> u32 {
        unsafe {
            self.count += 1;
            if self.head == u32::MAX {
                // allocate new
                let top = self.top;
                if self.count > self.capacity {
                    let new_capacity = (self.capacity * 2).max(16);
                    self.ptr = self
                        .storage
                        .resize(self.layout.repeat(new_capacity as usize).unwrap().0.size());
                    self.capacity = new_capacity;
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
    }
    pub fn free(&mut self, index: u32) {
        self.count -= 1;
        unsafe {
            let current_free_location = self.get_mut(index);

            // The first 4 bytes of the entry is populated with self.head
            *(current_free_location as *mut u32) = self.head;

            // All other bytes are zeroed
            let slice = std::slice::from_raw_parts_mut(current_free_location, self.layout.size());
            slice[std::mem::size_of::<u32>()..].fill(0);

            // push to freelist
            self.head = index;
        }
    }

    #[inline]
    pub unsafe fn get(&self, ptr: u32) -> *const u8 {
        unsafe { self.ptr.byte_add(self.layout.size() * ptr as usize) }
    }
    #[inline]
    pub unsafe fn get_mut(&mut self, ptr: u32) -> *mut u8 {
        unsafe { self.ptr.byte_add(self.layout.size() * ptr as usize) }
    }

    #[inline]
    pub unsafe fn get_item<T>(&self, ptr: u32) -> &T {
        unsafe {
            debug_assert_eq!(Layout::new::<T>().pad_to_align(), self.layout);
            &*(self.ptr.byte_add(Layout::new::<T>().pad_to_align().size() * ptr as usize) as *const T)
        }
    }
    #[inline]
    pub unsafe fn get_item_mut<T>(&mut self, ptr: u32) -> &mut T {
        unsafe {
            debug_assert_eq!(Layout::new::<T>().pad_to_align(), self.layout);
            &mut *(self.ptr.byte_add(Layout::new::<T>().pad_to_align().size() * ptr as usize) as *mut T)
        }
    }

    /// Total number of elements that are either occupied, or are marked invalid
    pub fn used_capacity(&self) -> u32 {
        self.top
    }
}
