use std::{
    alloc::{alloc, Layout},
    collections::BTreeMap,
    marker::PhantomData,
    mem::MaybeUninit,
};

use rhyolite::{
    ash::{
        prelude::VkResult,
        vk::{self, Handle},
    },
    vk_mem::{Alloc, Allocation, AllocationCreateInfo},
    HasDevice, PhysicalDeviceMemoryModel,
};

use crate::BitMask;

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
    num_items_per_chunk: u32,
    chunks: Vec<*mut u8>,

    count: u32,
    gpu_pool: Option<GPUPool>,
}

struct GPUPool {
    allocator: rhyolite::Allocator,
    device_allocations: Vec<Allocation>,
    device_buffer: vk::Buffer,
    device_address: u64,
    device_buffer_memory_requirements: vk::MemoryRequirements,
    host_allocations: Vec<(vk::Buffer, Allocation)>,
    /// 0..num_chunks_to_bind is bound to memory
    /// num_chunks_to_bind.. is not bound
    num_chunks_to_bind: u32,
    change_tracker: Option<PoolChangeTracker>,
}
impl Drop for GPUPool {
    fn drop(&mut self) {
        unsafe {
            self.allocator
                .free_memory_pages(&mut self.device_allocations);
            self.allocator
                .device()
                .destroy_buffer(self.device_buffer, None);

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
    pub fn new(layout: Layout, chunk_size: usize) -> Self {
        let num_items_per_chunk = (chunk_size / layout.pad_to_align().size()) as u32;
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
        min_chunk_size: usize,
        allocator: rhyolite::Allocator,
        max_size: u64,
        mut usage: vk::BufferUsageFlags,
    ) -> VkResult<Self> {
        if allocator
            .device()
            .physical_device()
            .properties()
            .memory_model
            .storage_buffer_should_use_staging()
        {
            usage |= vk::BufferUsageFlags::TRANSFER_DST;
        }
        let device_buffer = unsafe {
            allocator.device().create_buffer(
                &vk::BufferCreateInfo {
                    flags: vk::BufferCreateFlags::SPARSE_RESIDENCY
                        | vk::BufferCreateFlags::SPARSE_BINDING,
                    size: max_size,
                    usage,
                    ..Default::default()
                },
                None,
            )?
        };
        let device_buffer_memory_requirements = unsafe {
            allocator
                .device()
                .get_buffer_memory_requirements(device_buffer)
        };
        let device_address = if usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
            unsafe {
                allocator
                    .device()
                    .get_buffer_device_address(&vk::BufferDeviceAddressInfo {
                        buffer: device_buffer,
                        ..Default::default()
                    })
            }
        } else {
            0
        };
        let mut pool = Self::new(
            layout,
            min_chunk_size.max(device_buffer_memory_requirements.alignment as usize),
        );
        pool.gpu_pool = Some(GPUPool {
            change_tracker: if allocator
                .device()
                .physical_device()
                .properties()
                .memory_model
                .storage_buffer_should_use_staging()
            {
                Some(PoolChangeTracker {
                    tree: BTreeMap::new(),
                })
            } else {
                None
            },
            device_allocations: Vec::new(),
            host_allocations: Vec::new(),
            device_buffer,
            device_address,
            device_buffer_memory_requirements,
            num_chunks_to_bind: 0,
            allocator,
        });
        Ok(pool)
    }
    pub fn device_address(&self) -> vk::DeviceAddress {
        self.gpu_pool
            .as_ref()
            .map(|x| x.device_address)
            .unwrap_or(0)
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
            let chunk_index = top / self.num_items_per_chunk;
            if chunk_index as usize >= self.chunks.len() {
                // allocate new block
                self.alloc_new_chunk().unwrap();
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
    unsafe fn alloc_new_chunk(&mut self) -> VkResult<()> {
        if let Some(gpu_pool) = self.gpu_pool.as_mut() {
            let ptr = if gpu_pool
                .allocator
                .device()
                .physical_device()
                .properties()
                .memory_model
                .storage_buffer_should_use_staging()
            {
                assert!(gpu_pool.change_tracker.is_some());
                let device_allocation = gpu_pool.allocator.allocate_memory(
                    &vk::MemoryRequirements {
                        size: self.chunk_size,
                        ..gpu_pool.device_buffer_memory_requirements
                    },
                    &AllocationCreateInfo {
                        usage: rhyolite::vk_mem::MemoryUsage::Unknown,
                        required_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                        ..Default::default()
                    },
                )?;

                let (host_buffer, mut host_allocation) = gpu_pool.allocator.create_buffer(
                    &vk::BufferCreateInfo {
                        size: self.chunk_size,
                        usage: vk::BufferUsageFlags::TRANSFER_SRC,
                        ..Default::default()
                    },
                    &AllocationCreateInfo {
                        flags: rhyolite::vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM,
                        required_flags: vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_CACHED,
                        ..Default::default()
                    },
                )?;
                let ptr = gpu_pool.allocator.map_memory(&mut host_allocation)?;
                gpu_pool.device_allocations.push(device_allocation);
                gpu_pool
                    .host_allocations
                    .push((host_buffer, host_allocation));
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
                        required_flags: vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_CACHED,
                        preferred_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                        ..Default::default()
                    },
                )?;
                let ptr = gpu_pool.allocator.map_memory(&mut allocation)?;
                gpu_pool.device_allocations.push(allocation);
                ptr
            };
            assert!(!ptr.is_null());
            self.chunks.push(ptr);
            gpu_pool.num_chunks_to_bind += 1;
        } else {
            let layout =
                Layout::from_size_align(self.chunk_size as usize, self.layout.align()).unwrap();
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
        let chunk_index = ptr / self.num_items_per_chunk;
        let item_index = ptr - chunk_index * self.num_items_per_chunk;
        return self
            .chunks
            .get_unchecked(chunk_index as usize)
            .add(item_index as usize * self.layout.size());
    }
    #[inline]
    pub unsafe fn get_mut(&mut self, ptr: u32) -> *mut u8 {
        if let Some(gpu_pool) = &mut self.gpu_pool {
            if let Some(change_tracker) = &mut gpu_pool.change_tracker {
                change_tracker.set(ptr);
            }
        }
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

    pub(crate) fn bind_sparse(
        &mut self,
    ) -> (
        vk::Buffer,
        impl ExactSizeIterator<Item = vk::SparseMemoryBind> + '_,
    ) {
        let num_chunks_to_bind = self
            .gpu_pool
            .as_ref()
            .map(|x| x.num_chunks_to_bind)
            .unwrap_or(0);
        let chunk_size = self.chunk_size;
        let buffer = self
            .gpu_pool
            .as_ref()
            .map(|x| x.device_buffer)
            .unwrap_or_default();
        let (chunk_allocations, allocator) = self
            .gpu_pool
            .as_mut()
            .map(|x| (x.device_allocations.as_mut_slice(), Some(&x.allocator)))
            .unwrap_or((&mut [], None));
        let num_skips = chunk_allocations.len() - num_chunks_to_bind as usize;
        let iter = chunk_allocations
            .iter_mut()
            .enumerate()
            .skip(num_skips)
            .map(move |(i, chunk)| {
                let allocation = allocator.unwrap().get_allocation_info(chunk);
                vk::SparseMemoryBind {
                    resource_offset: i as u64 * chunk_size,
                    size: chunk_size,
                    memory: allocation.device_memory,
                    memory_offset: allocation.offset,
                    flags: vk::SparseMemoryBindFlags::empty(),
                }
            });
        (buffer, iter)
    }

    pub(crate) fn iter_changes(
        &self,
    ) -> impl Iterator<Item = (vk::Buffer, vk::Buffer, Vec<vk::BufferCopy>)> + '_ {
        let iter = self
            .gpu_pool
            .as_ref()
            .and_then(|x| x.change_tracker.as_ref())
            .unwrap_or(EMPTY_TRACKER)
            .iter();

        let mut iter = iter.map(|i| {
            let chunk_index = i / self.num_items_per_chunk;
            let item_index = i - chunk_index * self.num_items_per_chunk;
            let gpu_pool = self.gpu_pool.as_ref().unwrap();
            let src = gpu_pool.host_allocations[chunk_index as usize].0;
            let dst = gpu_pool.device_buffer;
            let region = vk::BufferCopy {
                src_offset: item_index as u64 * self.layout.size() as u64,
                dst_offset: i as u64 * self.layout.size() as u64,
                size: self.layout.size() as u64,
            };
            (src, dst, region)
        });
        (0..).scan(
            (vk::Buffer::null(), vk::Buffer::null(), Vec::new()),
            move |(current_src, current_dst, current_regions), _| {
                while let Some((src, dst, region)) = iter.next() {
                    if current_src.is_null() {
                        // Initialize
                        *current_src = src;
                        *current_dst = dst;
                        current_regions.push(region);
                    } else if src != *current_src || dst != *current_dst {
                        // Reinitialize
                        let ret = (*current_src, *current_dst, std::mem::take(current_regions));
                        *current_src = src;
                        *current_dst = dst;
                        current_regions.push(region);
                        return Some(ret);
                    } else {
                        let last_region = current_regions.last_mut().unwrap();
                        if last_region.src_offset + last_region.size == region.src_offset
                            && last_region.dst_offset + last_region.size == region.dst_offset
                        {
                            // Extend
                            last_region.size += region.size;
                        } else {
                            // Append
                            current_regions.push(region);
                        }
                    }
                }
                if current_regions.is_empty() {
                    None
                } else {
                    assert!(!current_src.is_null());
                    assert!(!current_dst.is_null());
                    Some((*current_src, *current_dst, std::mem::take(current_regions)))
                }
            },
        )
    }
    pub(crate) fn clear_changes(&mut self) {
        if let Some(tracker) = self
            .gpu_pool
            .as_mut()
            .and_then(|x| x.change_tracker.as_mut())
        {
            tracker.clear();
        }
    }
}
const EMPTY_TRACKER: &PoolChangeTracker = &PoolChangeTracker {
    tree: BTreeMap::new(),
};

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
                let layout =
                    Layout::from_size_align(self.chunk_size as usize, self.layout.align()).unwrap();
                for chunk in self.chunks.iter() {
                    let chunk = *chunk;
                    std::alloc::dealloc(chunk, layout);
                }
            }
        }
    }
}

#[derive(Default)]
struct PoolChangeTracker {
    tree: BTreeMap<u32, BitMask<512>>,
}
impl PoolChangeTracker {
    fn set(&mut self, index: u32) {
        let chunk_index = index / 512;
        let bit_index = index - chunk_index * 512;
        self.tree
            .entry(chunk_index)
            .or_default()
            .set(bit_index as usize, true);
    }
    fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.tree.iter().flat_map(|(chunk_index, bitmask)| {
            bitmask
                .iter_set_bits()
                .map(move |bit_index| chunk_index * 512 + bit_index as u32)
        })
    }
    fn clear(&mut self) {
        self.tree.clear();
    }
}
