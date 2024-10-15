use std::{alloc::{alloc, Layout}, marker::PhantomData, mem::MaybeUninit};

use rhyolite::{ash::{prelude::VkResult, vk}, vk_mem::{Alloc, Allocation, AllocationCreateInfo}, HasDevice, PhysicalDeviceMemoryModel};

pub struct Pool {
    /// Size of one individual allocation
    layout: Layout,
    /// Head of freelist
    head: u32,

    /// Top of free items.
    top: u32,
    /// Number of items to request when we run out of space.
    /// When running out of space, request chunk_size bytes.
    chunk_size: u64,
    /// Log2 of number of items in a chunk
    num_items_per_chunk: usize,
    chunks: Vec<*mut u8>,

    count: u32,
    gpu_pool: Option<GPUPool>,
}

struct GPUPool {
    allocator: rhyolite::Allocator,
    device_allocations: Vec<Allocation>,
    device_buffer: vk::Buffer,
    device_buffer_memory_requirements: vk::MemoryRequirements,
    host_allocations: Vec<(vk::Buffer, Allocation)>,
    /// 0..num_chunks_to_bind is bound to memory
    /// num_chunks_to_bind.. is not bound
    num_chunks_to_bind: u32,
}
impl Drop for GPUPool {
    fn drop(&mut self) {
        unsafe {
            self.allocator.free_memory_pages(&mut self.device_allocations);
            self.allocator.device().destroy_buffer(self.device_buffer, None);

            for (buffer, mut allocation) in self.host_allocations.drain(..) {
                self.allocator.unmap_memory(&mut allocation);
                self.allocator.destroy_buffer(buffer, &mut allocation);
            }
        }
    }
}

unsafe impl Send for Pool {}
unsafe impl Sync for Pool {}

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
    pub fn new(layout: Layout, num_items_per_chunk: usize) -> Self {
        let chunk_size = layout.pad_to_align().size() * (1 << num_items_per_chunk);
        Self {
            layout: layout.pad_to_align(),
            head: u32::MAX,
            top: 0,
            chunk_size: chunk_size as u64,
            chunks: Vec::new(),
            count: 0,
            num_items_per_chunk,
            gpu_pool: None,
        }
    }
    pub fn new_gpu_pool(
        layout: Layout,
        min_num_items_per_chunk: usize,
        allocator: rhyolite::Allocator,
        max_size: u64,
        mut usage: vk::BufferUsageFlags,
    ) -> VkResult<Self> {
        if allocator.device().physical_device().properties().memory_model.storage_buffer_should_use_staging() {
            usage |= vk::BufferUsageFlags::TRANSFER_DST;
        }
        let device_buffer = unsafe {
            allocator.device().create_buffer(&vk::BufferCreateInfo {
                flags: vk::BufferCreateFlags::SPARSE_RESIDENCY | vk::BufferCreateFlags::SPARSE_BINDING,
                size: max_size,
                usage,
                ..Default::default()
            }, None)?
        };
        let device_buffer_memory_requirements = unsafe {
            allocator.device().get_buffer_memory_requirements(device_buffer)
        };
        let mut pool = Self::new(layout, min_num_items_per_chunk);
        // TODO: ensure that the chunk size plays well with the buffer alignment requirments.
        pool.gpu_pool = Some(GPUPool {
            device_allocations: Vec::new(),
            host_allocations: Vec::new(),
            device_buffer,
            device_buffer_memory_requirements,
            num_chunks_to_bind: 0,
            allocator,
        });
        Ok(pool)
    }
    pub fn count(&self) -> u32 {
        self.count
    }
    pub unsafe fn alloc<T: Default>(&mut self) -> u32 {
        debug_assert_eq!(Layout::new::<T>(), self.layout);
        let ptr = self.alloc_uninitialized();
        let item = self.get_item_mut::<T>(ptr);
        *item = T::default();
        ptr
    }
    pub unsafe fn alloc_uninitialized(&mut self) -> u32 {
        self.count += 1;
        if self.head == u32::MAX {
            // allocate new
            let top = self.top;
            let chunk_index = top as usize >> self.num_items_per_chunk;
            if chunk_index >= self.chunks.len() {
                // allocate new block
                self.alloc_new_chunk();
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
    unsafe fn alloc_new_chunk(&mut self) -> VkResult<()>{
        if let Some(gpu_pool) = self.gpu_pool.as_mut() {
            let ptr = if gpu_pool.allocator.device().physical_device().properties().memory_model.storage_buffer_should_use_staging() {
                let device_allocation = gpu_pool.allocator.allocate_memory(
                    &vk::MemoryRequirements {
                        size: self.chunk_size,
                        ..gpu_pool.device_buffer_memory_requirements
                    },
                    &AllocationCreateInfo {
                        usage: rhyolite::vk_mem::MemoryUsage::Unknown,
                        required_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    ..Default::default()
                })?;
                
                let (host_buffer, mut host_allocation) = gpu_pool.allocator.create_buffer(&vk::BufferCreateInfo {
                    size: self.chunk_size,
                    usage: vk::BufferUsageFlags::TRANSFER_SRC,
                    ..Default::default()
                }, &AllocationCreateInfo {
                    flags: rhyolite::vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM,
                    required_flags: vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_CACHED,
                    ..Default::default()
                })?;
                let ptr = gpu_pool.allocator.map_memory(&mut host_allocation)?;
                gpu_pool.device_allocations.push(device_allocation);
                gpu_pool.host_allocations.push((host_buffer, host_allocation));
                ptr

            } else {
                let mut allocation = gpu_pool.allocator.allocate_memory(
                    &vk::MemoryRequirements {
                        size: self.chunk_size,
                        ..gpu_pool.device_buffer_memory_requirements
                    },
                    &AllocationCreateInfo {
                        usage: rhyolite::vk_mem::MemoryUsage::Unknown,
                        flags: rhyolite::vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM,
                        required_flags: vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_CACHED,
                        preferred_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    ..Default::default()
                })?;
                let ptr = gpu_pool.allocator.map_memory(&mut allocation)?;
                gpu_pool.device_allocations.push(allocation);
                ptr
            };
            assert!(!ptr.is_null());
            self.chunks.push(ptr);
            gpu_pool.num_chunks_to_bind += 1;
        } else {
            let layout = Layout::from_size_align(self.chunk_size as usize, self.layout.align()).unwrap();
            let block = std::alloc::alloc_zeroed(layout);
            self.chunks.push(block);
        }
        Ok(())
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

    pub fn num_chunks(&self) -> usize {
        self.chunks.len()
    }

    #[inline]
    pub unsafe fn get(&self, ptr: u32) -> *const u8 {
        let chunk_index = (ptr as usize) >> self.num_items_per_chunk;
        let item_index = (ptr as usize) & ((1 << self.num_items_per_chunk) - 1);
        return self
            .chunks
            .get_unchecked(chunk_index)
            .add(item_index * self.layout.size());
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

    pub fn iter_entries<T>(&self) -> PoolIterator<T> {
        debug_assert_eq!(Layout::new::<T>().pad_to_align(), self.layout);
        PoolIterator {
            pool: self,
            cur: 0,
            _marker: PhantomData,
        }
    }

    pub(crate) fn bind_sparse(&mut self) -> (vk::Buffer, impl ExactSizeIterator<Item = vk::SparseMemoryBind> + '_) {
        let num_chunks_to_bind = self.gpu_pool.as_ref().map(|x| x.num_chunks_to_bind).unwrap_or(0);
        let chunk_size = self.chunk_size;
        let buffer = self.gpu_pool.as_ref().map(|x| x.device_buffer).unwrap_or_default();
        let (chunk_allocations, allocator) = self.gpu_pool.as_mut().map(|x| (x.device_allocations.as_mut_slice(), Some(&x.allocator))).unwrap_or((&mut [], None));
        let num_skips = chunk_allocations.len() - num_chunks_to_bind as usize;
        let iter = chunk_allocations.iter_mut().enumerate().skip(num_skips).map(move |(i, chunk)| {
            let allocation = allocator.unwrap().get_allocation_info(chunk);
            vk::SparseMemoryBind {
                resource_offset: i as u64 * chunk_size,
                size: chunk_size,
                memory: allocation.device_memory,
                memory_offset: allocation.offset,
                flags: vk::SparseMemoryBindFlags::empty(),
            }
        });
        (buffer,iter)
    }
}

pub struct PoolIterator<'a, T> {
    pool: &'a Pool,
    cur: u32,
    _marker: PhantomData<T>,
}

impl<'a, T: 'a> Iterator for PoolIterator<'a, T> {
    type Item = &'a MaybeUninit<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur >= self.pool.top {
            return None;
        }
        let item: &'a MaybeUninit<T> = unsafe {
            let item = self.pool.get(self.cur);
            std::mem::transmute(item)
        };
        self.cur += 1;
        Some(item)
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        if let Some(gpu_pool) = self.gpu_pool.take() {
            drop(gpu_pool);
        } else {
            // CPU Pool. Drop all chunks using host allocator.
            unsafe {
                let layout = Layout::from_size_align(self.chunk_size as usize, self.layout.align()).unwrap();
                for chunk in self.chunks.iter() {
                    let chunk = *chunk;
                    std::alloc::dealloc(chunk, layout);
                }
            }
        }
    }
}
