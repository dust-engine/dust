use nohash::IntMap;
use std::{alloc::Layout, any::Any, collections::hash_map::Entry, sync::Arc};

pub trait PoolStorage: Send + Sync + Any {
    fn resize(&mut self, size: usize) -> *mut u8;
    fn device_address(&self) -> u64;
    /// Capture a read-only handle that keeps the *current* backing allocation
    /// alive, independently of any future [`PoolStorage::resize`] of this
    /// storage. Used by tree snapshots: the live pool may keep growing (and
    /// move its contents to a new allocation) while captures continue reading
    /// the allocation they pinned.
    fn capture(&self) -> FrozenStorage;
}

/// The storage behind a captured [`Pool`] (see [`Pool::capture`]): pins one
/// backing allocation for reading and refuses to grow.
pub struct FrozenStorage {
    device_address: u64,
    buffer: Option<Arc<dyn Any + Send + Sync>>,
}
impl FrozenStorage {
    pub fn new(device_address: u64, buffer: Option<Arc<dyn Any + Send + Sync>>) -> Self {
        Self {
            device_address,
            buffer,
        }
    }
}
impl PoolStorage for FrozenStorage {
    fn resize(&mut self, _size: usize) -> *mut u8 {
        panic!("captured pools are read-only and never grow")
    }
    fn device_address(&self) -> u64 {
        self.device_address
    }
    fn capture(&self) -> FrozenStorage {
        FrozenStorage {
            device_address: self.device_address,
            buffer: self.buffer.clone(),
        }
    }
}

/// A heap allocation that stays alive for as long as anything (the live pool,
/// or a [`FrozenStorage`] captured from it) still references it.
struct OwnedBuffer {
    ptr: *mut u8,
    layout: Layout,
}
unsafe impl Send for OwnedBuffer {}
unsafe impl Sync for OwnedBuffer {}
impl Drop for OwnedBuffer {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.ptr, self.layout) }
    }
}

pub struct DefaultPoolStorage {
    align: usize,
    buffer: Option<Arc<OwnedBuffer>>,
}
impl PoolStorage for DefaultPoolStorage {
    fn device_address(&self) -> u64 {
        0
    }
    fn resize(&mut self, size: usize) -> *mut u8 {
        unsafe {
            let layout = Layout::from_size_align_unchecked(size, self.align);
            let ptr = std::alloc::alloc(layout);
            if let Some(old) = self.buffer.take() {
                std::ptr::copy_nonoverlapping(old.ptr, ptr, old.layout.size().min(size));
                // The old allocation is deallocated here — unless a snapshot
                // captured it, in which case it lives until that capture drops.
            }
            self.buffer = Some(Arc::new(OwnedBuffer { ptr, layout }));
            ptr
        }
    }
    fn capture(&self) -> FrozenStorage {
        FrozenStorage {
            device_address: 0,
            buffer: self.buffer.as_ref().map(|x| x.clone() as Arc<dyn Any + Send + Sync>),
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

    /// Slots referenced by more than one parent edge, keyed by slot index.
    /// A slot absent from this map has exactly one parent edge (refcount 1);
    /// refcount 0 is not a representable state — the slot is freed instead.
    /// Allocation and freeing never touch this map: fresh slots start uniquely
    /// owned, and a slot may only be freed once it is uniquely owned again.
    pub(crate) refcounts: IntMap<u32, u32>,
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
                align: layout.align(),
                buffer: None,
            }),
            refcounts: IntMap::default(),
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
            refcounts: IntMap::default(),
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

    /// Record one additional parent edge to `slot`.
    ///
    /// Together with [`Pool::release`] this maintains a sparse per-slot
    /// reference count of incoming parent edges, used for copy-on-write
    /// structural sharing between tree versions.
    /// ```
    /// use std::alloc::Layout;
    /// use dust_vdb::pool::Pool;
    /// unsafe {
    ///   let mut pool = Pool::new(Layout::new::<u64>());
    ///   let slot = pool.alloc::<u64>();
    ///   assert!(!pool.is_shared(slot));
    ///   pool.retain(slot); // a second parent now references the slot
    ///   assert!(pool.is_shared(slot));
    ///   assert!(!pool.release(slot)); // back to one parent; still alive
    ///   assert!(pool.release(slot)); // last edge removed: caller must free
    ///   pool.free(slot);
    /// }
    /// ```
    pub fn retain(&mut self, slot: u32) {
        debug_assert!(slot < self.top);
        *self.refcounts.entry(slot).or_insert(1) += 1;
    }

    /// Whether `slot` is referenced by more than one parent edge. Shared slots
    /// are frozen: they must be copied (see [`Pool::copy_item`]) instead of
    /// mutated in place.
    #[inline]
    pub fn is_shared(&self, slot: u32) -> bool {
        self.refcounts.contains_key(&slot)
    }

    /// Remove one parent edge to `slot`. Returns `true` if that was the last
    /// edge: the slot is then unreachable, and the caller is responsible for
    /// releasing its children and freeing it.
    #[must_use]
    pub fn release(&mut self, slot: u32) -> bool {
        match self.refcounts.entry(slot) {
            Entry::Occupied(mut entry) => {
                let refcount = entry.get_mut();
                debug_assert!(*refcount >= 2);
                *refcount -= 1;
                if *refcount == 1 {
                    entry.remove();
                }
                false
            }
            Entry::Vacant(_) => true,
        }
    }

    /// A read-only copy of this pool sharing the current backing allocation,
    /// for use by tree snapshots. The capture keeps the allocation alive: the
    /// live pool may keep growing (moving its contents to a new allocation)
    /// without invalidating it.
    ///
    /// Reads through the capture are only meaningful for slots frozen by
    /// copy-on-write sharing: those bytes are never written again while any
    /// version references them, which is what makes cross-thread snapshot
    /// reads sound while the live pool is mutated concurrently.
    pub(crate) fn capture(&self) -> Pool {
        Pool {
            layout: self.layout,
            head: u32::MAX,
            top: self.top,
            count: self.count,
            capacity: self.capacity,
            ptr: self.ptr,
            storage: Box::new(self.storage.capture()),
            refcounts: IntMap::default(),
        }
    }

    /// Allocate a new slot initialized with a copy of the item at `src`.
    ///
    /// Safety: `src` must point to a live item of type `T` in this pool.
    pub unsafe fn copy_item<T: Clone>(&mut self, src: u32) -> u32 {
        unsafe {
            debug_assert_eq!(Layout::new::<T>().pad_to_align(), self.layout);
            let dst = self.alloc_uninitialized();
            // alloc_uninitialized may move the storage; form pointers only now.
            let src = self.get(src) as *const T;
            let dst_ptr = self.get_mut(dst) as *mut T;
            std::ptr::write(dst_ptr, (*src).clone());
            dst
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
            &*(self
                .ptr
                .byte_add(Layout::new::<T>().pad_to_align().size() * ptr as usize)
                as *const T)
        }
    }
    #[inline]
    pub unsafe fn get_item_mut<T>(&mut self, ptr: u32) -> &mut T {
        unsafe {
            debug_assert_eq!(Layout::new::<T>().pad_to_align(), self.layout);
            &mut *(self
                .ptr
                .byte_add(Layout::new::<T>().pad_to_align().size() * ptr as usize)
                as *mut T)
        }
    }

    /// Total number of elements that are either occupied, or are marked invalid
    pub fn used_capacity(&self) -> u32 {
        self.top
    }
}
