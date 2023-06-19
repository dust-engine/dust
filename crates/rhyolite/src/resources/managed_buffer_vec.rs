use std::{
    alloc::Layout,
    collections::{BTreeMap, BTreeSet},
};

use crate::{
    copy_buffer, copy_buffer_regions,
    future::{
        use_shared_state_with_old, GPUCommandFuture, PerFrameContainer, PerFrameState, RenderData,
        RenderRes, SharedDeviceState, SharedDeviceStateHostContainer,
    },
    utils::{either::Either, merge_ranges::MergeRangeIteratorExt},
    Allocator, BufferLike, HasDevice, ResidentBuffer,
};
use ash::vk;
use rhyolite::macros::commands;

type ManagedBufferVecInner =
    Either<PerFrameContainer<ResidentBuffer>, SharedDeviceState<ResidentBuffer>>;

pub enum ManagedBufferVec<T> {
    DirectWrite(ManagedBufferVecStrategyDirectWrite<T>),
    StagingBuffer(ManagedBufferVecStrategyStaging<T>),
}
impl<T> HasDevice for ManagedBufferVec<T> {
    fn device(&self) -> &std::sync::Arc<crate::Device> {
        match self {
            Self::DirectWrite(a) => a.allocator.device(),
            Self::StagingBuffer(a) => a.allocator.device(),
        }
    }
}

impl<T> ManagedBufferVec<T> {
    pub fn new(
        allocator: Allocator,
        buffer_usage_flags: vk::BufferUsageFlags,
        alignment: u32,
    ) -> Self {
        use crate::PhysicalDeviceMemoryModel::*;
        match allocator.physical_device().memory_model() {
            ResizableBar | UMA => Self::DirectWrite(ManagedBufferVecStrategyDirectWrite::new(
                allocator,
                buffer_usage_flags,
                alignment,
            )),
            Discrete | Bar => Self::StagingBuffer(ManagedBufferVecStrategyStaging::new(
                allocator,
                buffer_usage_flags,
                alignment,
            )),
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Self::DirectWrite(strategy) => strategy.len(),
            Self::StagingBuffer(strategy) => strategy.len(),
        }
    }
    pub fn allocator(&self) -> &Allocator {
        match self {
            Self::DirectWrite(strategy) => strategy.allocator(),
            Self::StagingBuffer(strategy) => strategy.allocator(),
        }
    }
    pub fn push(&mut self, item: T) {
        match self {
            Self::DirectWrite(strategy) => strategy.push(item),
            Self::StagingBuffer(strategy) => strategy.push(item),
        }
    }
    pub fn set(&mut self, index: usize, item: T) {
        match self {
            Self::DirectWrite(strategy) => strategy.set(index, item),
            Self::StagingBuffer(strategy) => strategy.set(index, item),
        }
    }

    pub fn buffer(
        &mut self,
    ) -> Option<impl GPUCommandFuture<Output = RenderRes<ManagedBufferVecInner>>> {
        let buffer = match self {
            Self::DirectWrite(strategy) => strategy.buffer().map(|b| Either::Left(b)),
            Self::StagingBuffer(strategy) => strategy.buffer().map(|b| Either::Right(b)),
        }?;

        let fut = commands! {
            match buffer {
                Either::Left(buffer) => {
                    RenderRes::new(Either::Left(buffer))
                }
                Either::Right(future) => {
                    let result = future.await;
                    result.map(|a| Either::Right(a))
                }
            }
        };
        Some(fut)
    }
}

pub enum ManagedBufferVecUnsized {
    DirectWrite(ManagedBufferVecStrategyDirectWriteUnsized),
    StagingBuffer(ManagedBufferVecStrategyStagingUnsized),
}
impl HasDevice for ManagedBufferVecUnsized {
    fn device(&self) -> &std::sync::Arc<crate::Device> {
        match self {
            Self::DirectWrite(a) => a.allocator.device(),
            Self::StagingBuffer(a) => a.allocator.device(),
        }
    }
}

impl ManagedBufferVecUnsized {
    pub fn new(
        allocator: Allocator,
        buffer_usage_flags: vk::BufferUsageFlags,
        layout: Layout,
        base_alignment: usize,
    ) -> Self {
        use crate::PhysicalDeviceMemoryModel::*;
        match allocator.physical_device().memory_model() {
            ResizableBar | UMA => {
                Self::DirectWrite(ManagedBufferVecStrategyDirectWriteUnsized::new(
                    allocator,
                    buffer_usage_flags,
                    layout,
                    base_alignment,
                ))
            }
            Discrete | Bar => Self::StagingBuffer(ManagedBufferVecStrategyStagingUnsized::new(
                allocator,
                buffer_usage_flags,
                layout,
                base_alignment,
            )),
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Self::DirectWrite(strategy) => strategy.len(),
            Self::StagingBuffer(strategy) => strategy.len(),
        }
    }
    pub fn allocator(&self) -> &Allocator {
        match self {
            Self::DirectWrite(strategy) => strategy.allocator(),
            Self::StagingBuffer(strategy) => strategy.allocator(),
        }
    }
    pub fn push(&mut self, item: &[u8]) {
        match self {
            Self::DirectWrite(strategy) => strategy.push(item),
            Self::StagingBuffer(strategy) => strategy.push(item),
        }
    }
    pub fn set(&mut self, index: usize, item: &[u8]) {
        match self {
            Self::DirectWrite(strategy) => strategy.set(index, item),
            Self::StagingBuffer(strategy) => strategy.set(index, item),
        }
    }

    pub fn buffer(
        &mut self,
    ) -> Option<impl GPUCommandFuture<Output = RenderRes<impl BufferLike + RenderData>>> {
        let buffer = match self {
            Self::DirectWrite(strategy) => strategy.buffer().map(|b| Either::Left(b)),
            Self::StagingBuffer(strategy) => strategy.buffer().map(|b| Either::Right(b)),
        }?;

        let fut = commands! {
            match buffer {
                Either::Left(buffer) => {
                    RenderRes::new(Either::Left(buffer))
                }
                Either::Right(future) => {
                    let result = future.await;
                    result.map(|a| Either::Right(a))
                }
            }
        };
        Some(fut)
    }
}

pub struct ManagedBufferVecStrategyDirectWrite<T> {
    allocator: Allocator,
    buffer_usage_flags: vk::BufferUsageFlags,
    buffers: PerFrameState<ResidentBuffer>,
    objects: Vec<T>,
    changeset: BTreeMap<vk::Buffer, BTreeSet<usize>>,
    alignment: u32,
}
impl<T> ManagedBufferVecStrategyDirectWrite<T> {
    pub fn new(
        allocator: Allocator,
        buffer_usage_flags: vk::BufferUsageFlags,
        alignment: u32,
    ) -> Self {
        Self {
            allocator,
            buffer_usage_flags,
            buffers: Default::default(),
            objects: Vec::new(),
            changeset: Default::default(),
            alignment,
        }
    }
    pub fn len(&self) -> usize {
        self.objects.len()
    }
    pub fn allocator(&self) -> &Allocator {
        &self.allocator
    }
    pub fn push(&mut self, item: T) {
        let index = self.objects.len();
        self.objects.push(item);
        for changes in self.changeset.values_mut() {
            changes.insert(index);
        }
    }
    pub fn set(&mut self, index: usize, item: T) {
        self.objects[index] = item;
        for changes in self.changeset.values_mut() {
            changes.insert(index);
        }
    }

    pub fn buffer(&mut self) -> Option<PerFrameContainer<ResidentBuffer>> {
        let item_size = std::alloc::Layout::new::<T>().pad_to_align().size();
        if self.objects.is_empty() {
            return None;
        }
        let create_buffer = || {
            let create_buffer = self
                .allocator
                .create_write_buffer_uninit_aligned(
                    (self.objects.capacity() * item_size) as u64,
                    self.buffer_usage_flags,
                    self.alignment as u64,
                )
                .unwrap();
            create_buffer.contents_mut().unwrap()[0..self.objects.len() * item_size]
                .copy_from_slice(unsafe {
                    std::slice::from_raw_parts(
                        self.objects.as_ptr() as *const u8,
                        self.objects.len() * item_size,
                    )
                });
            create_buffer
        };

        let buf = self
            .buffers
            .use_state(|| {
                let new_buffer = create_buffer();
                self.changeset
                    .insert(new_buffer.raw_buffer(), Default::default());
                new_buffer
            })
            .reuse(|buffer| {
                let changes = std::mem::take(self.changeset.get_mut(&buffer.raw_buffer()).unwrap());
                if buffer.size() < (self.objects.len() * item_size) as u64 {
                    // need to recreate the buffer
                    self.changeset.remove(&buffer.raw_buffer()).unwrap();
                    let new_buffer = create_buffer();
                    // Create empty entry for the buffer just created.
                    // From now on, changes need to be recorded for this new buffer.
                    self.changeset
                        .insert(new_buffer.raw_buffer(), Default::default());
                    *buffer = new_buffer;
                } else {
                    for (changed_index_start, num_changes) in changes.into_iter().merge_ranges() {
                        let start = changed_index_start * item_size;
                        let end = (changed_index_start + num_changes) * item_size;
                        buffer.contents_mut().unwrap()[start..end].copy_from_slice(unsafe {
                            std::slice::from_raw_parts(
                                (self.objects.as_ptr() as *const u8).add(start),
                                num_changes * item_size,
                            )
                        })
                    }
                }
            });
        Some(buf)
    }
}

pub struct ManagedBufferVecStrategyDirectWriteUnsized {
    base_alignment: usize,
    layout: Layout,
    allocator: Allocator,
    buffer_usage_flags: vk::BufferUsageFlags,
    buffers: PerFrameState<ResidentBuffer>,
    objects_buffer: Vec<u8>,
    num_items: usize,
    changeset: BTreeMap<vk::Buffer, BTreeSet<usize>>,
}
impl ManagedBufferVecStrategyDirectWriteUnsized {
    pub fn new(
        allocator: Allocator,
        buffer_usage_flags: vk::BufferUsageFlags,
        layout: Layout,
        base_alignment: usize,
    ) -> Self {
        Self {
            layout,
            allocator,
            buffer_usage_flags,
            buffers: Default::default(),
            objects_buffer: Vec::new(),
            changeset: Default::default(),
            num_items: 0,
            base_alignment,
        }
    }
    pub fn len(&self) -> usize {
        self.num_items
    }
    pub fn allocator(&self) -> &Allocator {
        &self.allocator
    }
    pub fn push(&mut self, item: &[u8]) {
        assert_eq!(item.len(), self.layout.size());

        let index = self.num_items;
        self.num_items += 1;
        self.objects_buffer
            .reserve(self.layout.pad_to_align().size());
        unsafe {
            let dst = self
                .objects_buffer
                .as_mut_ptr()
                .add(self.objects_buffer.len());
            let dst = std::slice::from_raw_parts_mut(dst, self.layout.pad_to_align().size());
            dst[0..self.layout.size()].copy_from_slice(item);
            dst[self.layout.size()..].fill(0);
            self.objects_buffer
                .set_len(self.objects_buffer.len() + self.layout.pad_to_align().size());
        }
        for changes in self.changeset.values_mut() {
            changes.insert(index);
        }
    }
    pub fn set(&mut self, index: usize, item: &[u8]) {
        assert_eq!(item.len(), self.layout.size());
        self.num_items = self.num_items.max(index + 1);

        let expected_len = self.num_items * self.layout.pad_to_align().size();
        self.objects_buffer
            .reserve(expected_len - self.objects_buffer.len());
        unsafe {
            let dst = self
                .objects_buffer
                .as_mut_ptr()
                .add(index * self.layout.pad_to_align().size());
            let dst = std::slice::from_raw_parts_mut(dst, self.layout.pad_to_align().size());
            dst[0..self.layout.size()].copy_from_slice(item);
            dst[self.layout.size()..].fill(0);
            self.objects_buffer.set_len(expected_len);
        }
        for changes in self.changeset.values_mut() {
            changes.insert(index);
        }
    }

    pub fn buffer(&mut self) -> Option<PerFrameContainer<ResidentBuffer>> {
        let item_size = self.layout.pad_to_align().size();
        if self.num_items == 0 {
            return None;
        }
        let create_buffer = || {
            let create_buffer = self
                .allocator
                .create_write_buffer_uninit_aligned(
                    self.objects_buffer.capacity() as u64,
                    self.buffer_usage_flags,
                    self.base_alignment as u64,
                )
                .unwrap();
            create_buffer.contents_mut().unwrap()[0..self.objects_buffer.len()]
                .copy_from_slice(&self.objects_buffer);
            create_buffer
        };

        let buf = self
            .buffers
            .use_state(|| {
                let new_buffer = create_buffer();
                self.changeset
                    .insert(new_buffer.raw_buffer(), Default::default());
                new_buffer
            })
            .reuse(|buffer| {
                let changes = std::mem::take(self.changeset.get_mut(&buffer.raw_buffer()).unwrap());
                if buffer.size() < self.objects_buffer.len() as u64 {
                    // need to recreate the buffer
                    self.changeset.remove(&buffer.raw_buffer()).unwrap();
                    let new_buffer = create_buffer();
                    // Create empty entry for the buffer just created.
                    // From now on, changes need to be recorded for this new buffer.
                    self.changeset
                        .insert(new_buffer.raw_buffer(), Default::default());
                    *buffer = new_buffer;
                } else {
                    for (changed_index_start, num_changes) in changes.into_iter().merge_ranges() {
                        let start = changed_index_start * item_size;
                        let end = (changed_index_start + num_changes) * item_size;
                        buffer.contents_mut().unwrap()[start..end].copy_from_slice(unsafe {
                            std::slice::from_raw_parts(
                                self.objects_buffer.as_ptr().add(start),
                                num_changes * item_size,
                            )
                        })
                    }
                }
            });
        Some(buf)
    }
}

pub struct ManagedBufferVecStrategyStaging<T> {
    allocator: Allocator,
    buffer_usage_flags: vk::BufferUsageFlags,

    device_buffer: Option<SharedDeviceStateHostContainer<ResidentBuffer>>,
    staging_buffer: PerFrameState<ResidentBuffer>,
    changes: BTreeMap<usize, T>,
    num_items: usize,
    alignment: u32,
}
impl<T> ManagedBufferVecStrategyStaging<T> {
    pub fn new(
        allocator: Allocator,
        buffer_usage_flags: vk::BufferUsageFlags,
        alignment: u32,
    ) -> Self {
        Self {
            allocator,
            buffer_usage_flags,
            device_buffer: None,
            staging_buffer: Default::default(),
            changes: Default::default(),
            alignment,
            num_items: 0,
        }
    }
    pub fn len(&self) -> usize {
        self.num_items
    }
    pub fn allocator(&self) -> &Allocator {
        &self.allocator
    }
    pub fn push(&mut self, item: T) {
        self.changes.insert(self.num_items, item);
        self.num_items += 1;
    }
    pub fn set(&mut self, index: usize, item: T) {
        self.changes.insert(index, item);
    }

    pub fn buffer(
        &mut self,
    ) -> Option<impl GPUCommandFuture<Output = RenderRes<SharedDeviceState<ResidentBuffer>>>> {
        let item_size = std::alloc::Layout::new::<T>().pad_to_align().size();
        if self.num_items == 0 {
            return None;
        }
        let staging_buffer_copy = if !self.changes.is_empty() {
            let expected_staging_size = (item_size * self.changes.len()) as u64;

            let staging_buffer = self
                .staging_buffer
                .use_state(|| {
                    self.allocator
                        .create_staging_buffer(expected_staging_size)
                        .unwrap()
                })
                .reuse(|old| {
                    if old.size() < expected_staging_size {
                        // Too small. Enlarge.
                        *old = self
                            .allocator
                            .create_staging_buffer(
                                (old.size() as u64 * 2).max(expected_staging_size),
                            )
                            .unwrap();
                    }
                });
            let changes = std::mem::take(&mut self.changes);
            let (changed_indices, changed_items): (Vec<usize>, Vec<T>) =
                changes.into_iter().unzip();
            staging_buffer
                .contents_mut()
                .unwrap()
                .copy_from_slice(unsafe {
                    std::slice::from_raw_parts(
                        changed_items.as_ptr() as *const u8,
                        std::mem::size_of_val(changed_items.as_slice()),
                    )
                });
            let staging_current_index = 0; // ???
            let buffer_copy = changed_indices
                .into_iter()
                .merge_ranges()
                .map(|(start, len)| vk::BufferCopy {
                    src_offset: staging_current_index * item_size as u64,
                    dst_offset: start as u64 * item_size as u64,
                    size: len as u64 * item_size as u64,
                })
                .collect::<Vec<_>>();
            Some((staging_buffer, buffer_copy))
        } else {
            None
        };

        let expected_whole_buffer_size = self.num_items as u64 * item_size as u64;
        let (device_buffer, old_device_buffer) = use_shared_state_with_old(
            &mut self.device_buffer,
            |_| {
                self.allocator
                    .create_device_buffer_uninit_aligned(
                        expected_whole_buffer_size,
                        self.buffer_usage_flags | vk::BufferUsageFlags::TRANSFER_DST,
                        self.alignment as u64,
                    )
                    .unwrap()
            },
            |buf| buf.size() < expected_whole_buffer_size,
        );

        let fut = commands! {
            let mut device_buffer = device_buffer;

            if let Some(old_buffer) = old_device_buffer {
                let old_buffer = old_buffer;
                copy_buffer(&old_buffer, &mut device_buffer).await;
                retain!(old_buffer);
            }
            if let Some((staging_buffer, buffer_copy)) = staging_buffer_copy {
                let staging_buffer = RenderRes::new(staging_buffer);
                copy_buffer_regions(&staging_buffer, &mut device_buffer, buffer_copy).await;
                retain!(staging_buffer);
            }

            device_buffer
        };
        Some(fut)
    }
}

pub struct ManagedBufferVecStrategyStagingUnsized {
    layout: Layout,
    allocator: Allocator,
    buffer_usage_flags: vk::BufferUsageFlags,

    device_buffer: Option<SharedDeviceStateHostContainer<ResidentBuffer>>,
    staging_buffer: PerFrameState<ResidentBuffer>,
    change_buffer: Vec<u8>,

    /// Mapping from raw index to index in change_buffer
    changes: BTreeMap<usize, usize>,
    num_items: usize,
    base_alignment: usize,
}
impl ManagedBufferVecStrategyStagingUnsized {
    pub fn new(
        allocator: Allocator,
        buffer_usage_flags: vk::BufferUsageFlags,
        layout: Layout,
        base_alignment: usize,
    ) -> Self {
        Self {
            layout,
            allocator,
            buffer_usage_flags,
            device_buffer: None,
            staging_buffer: Default::default(),
            change_buffer: Vec::new(),
            changes: Default::default(),
            num_items: 0,
            base_alignment,
        }
    }
    pub fn len(&self) -> usize {
        self.num_items
    }
    pub fn allocator(&self) -> &Allocator {
        &self.allocator
    }
    pub fn push(&mut self, item: &[u8]) {
        assert_eq!(item.len(), self.layout.size());
        let change_buffer_index = self.change_buffer.len() / self.layout.pad_to_align().size();
        self.change_buffer
            .reserve(self.layout.pad_to_align().size());
        unsafe {
            let dst = self
                .change_buffer
                .as_mut_ptr()
                .add(self.change_buffer.len());
            let dst = std::slice::from_raw_parts_mut(dst, self.layout.pad_to_align().size());
            dst[0..self.layout.size()].copy_from_slice(item);
            dst[self.layout.size()..].fill(0);
            self.change_buffer
                .set_len(self.change_buffer.len() + self.layout.pad_to_align().size());
        }

        self.changes.insert(self.num_items, change_buffer_index);
        self.num_items += 1;
    }
    pub fn set(&mut self, index: usize, item: &[u8]) {
        assert_eq!(item.len(), self.layout.size());
        self.num_items = self.num_items.max(index + 1);
        if let Some(existing_change_buffer_index) = self.changes.get(&index) {
            unsafe {
                let dst = self
                    .change_buffer
                    .as_mut_ptr()
                    .add(existing_change_buffer_index * self.layout.pad_to_align().size());
                let dst = std::slice::from_raw_parts_mut(dst, self.layout.pad_to_align().size());
                dst[0..self.layout.size()].copy_from_slice(item);
                dst[self.layout.size()..].fill(0);
            }
        } else {
            let change_buffer_index = self.change_buffer.len() / self.layout.pad_to_align().size();
            self.change_buffer
                .reserve(self.layout.pad_to_align().size());
            unsafe {
                let dst = self
                    .change_buffer
                    .as_mut_ptr()
                    .add(self.change_buffer.len());
                let dst = std::slice::from_raw_parts_mut(dst, self.layout.pad_to_align().size());
                dst[0..self.layout.size()].copy_from_slice(item);
                dst[self.layout.size()..].fill(0);
                self.change_buffer
                    .set_len(self.change_buffer.len() + self.layout.pad_to_align().size());
            }
            self.changes.insert(index, change_buffer_index);
        }
    }

    pub fn buffer(
        &mut self,
    ) -> Option<impl GPUCommandFuture<Output = RenderRes<SharedDeviceState<ResidentBuffer>>>> {
        let item_size = self.layout.pad_to_align().size();
        if self.num_items == 0 {
            return None;
        }
        let staging_buffer_copy = if !self.change_buffer.is_empty() {
            let expected_staging_size = self.change_buffer.len() as u64;
            let staging_buffer = self
                .staging_buffer
                .use_state(|| {
                    self.allocator
                        .create_staging_buffer(self.change_buffer.len() as u64)
                        .unwrap()
                })
                .reuse(|old| {
                    if old.size() < expected_staging_size {
                        // Too small. Enlarge.
                        *old = self
                            .allocator
                            .create_staging_buffer(
                                (old.size() as u64 * 2).max(expected_staging_size),
                            )
                            .unwrap();
                    }
                });

            let changes = std::mem::take(&mut self.changes);
            let ordered_change_buffer =
                &mut staging_buffer.contents_mut().unwrap()[0..self.change_buffer.len()];
            {
                let mut ordered_change_buffer_len = 0;
                for (_, change_buffer_index) in changes.iter() {
                    let src = &self.change_buffer
                        [change_buffer_index * item_size..(change_buffer_index + 1) * item_size];
                    let dst = &mut ordered_change_buffer
                        [ordered_change_buffer_len..ordered_change_buffer_len + item_size];
                    dst.copy_from_slice(src);
                    ordered_change_buffer_len += item_size;
                }
                self.change_buffer.clear();
                assert_eq!(ordered_change_buffer.len(), ordered_change_buffer_len);
            }

            let mut staging_current_index = 0;
            let buffer_copy = changes
                .keys()
                .cloned()
                .merge_ranges()
                .map(|(start, len)| {
                    let copy = vk::BufferCopy {
                        src_offset: staging_current_index * item_size as u64,
                        dst_offset: start as u64 * item_size as u64,
                        size: len as u64 * item_size as u64,
                    };
                    staging_current_index += len as u64;
                    copy
                })
                .collect::<Vec<_>>();
            Some((staging_buffer, buffer_copy))
        } else {
            None
        };

        let expected_whole_buffer_size = self.num_items as u64 * item_size as u64;
        let (device_buffer, old_device_buffer) = use_shared_state_with_old(
            &mut self.device_buffer,
            |_| {
                self.allocator
                    .create_device_buffer_uninit_aligned(
                        expected_whole_buffer_size,
                        self.buffer_usage_flags | vk::BufferUsageFlags::TRANSFER_DST,
                        self.base_alignment as u64,
                    )
                    .unwrap()
            },
            |buf| buf.size() < expected_whole_buffer_size,
        );

        let fut = commands! {
            let mut device_buffer = device_buffer;

            if let Some(old_buffer) = old_device_buffer {
                let old_buffer = old_buffer;
                copy_buffer(&old_buffer, &mut device_buffer).await;
                retain!(old_buffer);
            }
            if let Some((staging_buffer, buffer_copy)) = staging_buffer_copy {
                let staging_buffer = RenderRes::new(staging_buffer);
                copy_buffer_regions(&staging_buffer, &mut device_buffer, buffer_copy).await;
                retain!(staging_buffer);
            }

            device_buffer
        };
        Some(fut)
    }
}
