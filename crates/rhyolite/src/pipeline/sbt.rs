// Gives out SbtEntryHandle.
// User gives you the T. On duplicates, return exisitng handle.
// When unique, create new entry.

use std::{
    alloc::Layout,
    collections::{BTreeMap, BTreeSet, HashMap},
    hash::Hash,
    sync::{Arc, Weak},
};

use ash::{prelude::VkResult, vk};
use macros::commands;

use crate::{
    copy_buffer, copy_buffer_regions,
    future::{
        use_shared_state, GPUCommandFuture, GPUCommandFutureExt, PerFrameState, RenderRes,
        SharedDeviceStateHostContainer,
    },
    utils::merge_ranges::MergeRangeIteratorExt,
    Allocator, BufferLike, HasDevice, PhysicalDeviceMemoryModel, RayTracingPipeline,
    ResidentBuffer, SbtHandles,
};

pub trait HitgroupSbtEntry: Hash + Eq + Clone {
    type ShaderParameter: Copy;
    fn parameter(&self, raytype_index: u32) -> Self::ShaderParameter;
    fn hitgroup_index(&self, raytype_index: u32) -> usize;
}

struct HitgroupSbtHandleInner {
    id: usize,
    sender: std::sync::mpsc::Sender<usize>,
}
impl Drop for HitgroupSbtHandleInner {
    fn drop(&mut self) {
        self.sender.send(self.id).unwrap();
    }
}

// This is to be included on the component of entities.
#[derive(Clone)]
pub struct HitgroupSbtHandle(Arc<HitgroupSbtHandleInner>);

pub struct HitgroupSbtVec<T: HitgroupSbtEntry> {
    allocator: Allocator,
    layout: SbtLayout,
    total_raytype: u32,
    shader_group_handles: SbtHandles,
    /// A map of (item -> (id, generation, handle))
    handles: HashMap<T, (usize, Weak<HitgroupSbtHandleInner>)>,

    /// A list of (generation, item) indexed by id
    entries: Vec<T>,

    sender: std::sync::mpsc::Sender<usize>,
    available_indices: std::sync::mpsc::Receiver<usize>,

    changeset: BTreeMap<vk::Buffer, BTreeSet<usize>>,
    frames: PerFrameState<ResidentBuffer>,
    /// If None, `frames` are assumed to be device-local and are to be used directly by the GPU.
    device_buffer: Option<SharedDeviceStateHostContainer<ResidentBuffer>>,
}

impl<T: HitgroupSbtEntry> HasDevice for HitgroupSbtVec<T> {
    fn device(&self) -> &Arc<crate::Device> {
        self.allocator.device()
    }
}

impl<T: HitgroupSbtEntry> HitgroupSbtVec<T> {
    pub fn new(pipeline: &RayTracingPipeline, allocator: Allocator) -> Self {
        let (sender, receiver) = std::sync::mpsc::channel();
        let shader_group_handles = pipeline.get_shader_group_handles();
        let layout = SbtLayout::new::<T>(&shader_group_handles, 1);
        Self {
            allocator,
            total_raytype: 1,
            handles: HashMap::new(),
            entries: Vec::new(),
            sender,
            available_indices: receiver,
            changeset: Default::default(),
            frames: Default::default(),
            shader_group_handles,
            device_buffer: None,
            layout,
        }
    }
    pub fn get(&self, handle: &HitgroupSbtHandle) -> &T {
        &self.entries[handle.0.id]
    }
    pub fn add(&mut self, item: T) -> HitgroupSbtHandle {
        if let Some((id, retained_handle)) = self.handles.get_mut(&item) {
            // This item was previously requested
            if let Some(handle) = retained_handle.upgrade() {
                // And the handle is still valid now
                assert_eq!(*id, handle.id);
                return HitgroupSbtHandle(handle);
            } else if self.entries[*id] == item {
                // But the handle is no longer valid. Fortunately no one has overwritten the entry yet.
                // Let's reuse that entry.
                let handle = Arc::new(HitgroupSbtHandleInner {
                    id: *id,
                    sender: self.sender.clone(),
                });
                *retained_handle = Arc::downgrade(&handle);
                // This way, no need to call self.record_location_update
                return HitgroupSbtHandle(handle);
            }
            unreachable!();
        }

        loop {
            let candidate = if let Some(candidate) = self.available_indices.try_recv().ok() {
                candidate
            } else {
                // No more available indices. Need to create a new entry.
                break;
            };
            let prev_item = &self.entries[candidate];
            if self.handles[prev_item].1.strong_count() == 0 {
                // This slot is safe to reuse.
                let handle = Arc::new(HitgroupSbtHandleInner {
                    id: candidate,
                    sender: self.sender.clone(),
                });
                self.handles.remove(prev_item);
                self.entries[candidate] = item.clone();
                assert!(self
                    .handles
                    .insert(item, (candidate, Arc::downgrade(&handle)))
                    .is_none());
                self.record_location_update(candidate);
                return HitgroupSbtHandle(handle);
            } else {
                // This slot was already reused. Try again.
                continue;
            }
        }

        let candidate = self.entries.len();
        self.entries.push(item.clone());
        let handle = Arc::new(HitgroupSbtHandleInner {
            id: candidate,
            sender: self.sender.clone(),
        });
        assert!(self
            .handles
            .insert(item, (candidate, Arc::downgrade(&handle)))
            .is_none());
        self.record_location_update(candidate);
        return HitgroupSbtHandle(handle);
    }

    fn record_location_update(&mut self, location: usize) {
        for (_, changes) in self.changeset.iter_mut() {
            // Iterate over all device-owned buffers, and defer the changes
            changes.insert(location);
        }
    }
    pub fn get_sbt_buffer(
        &mut self,
    ) -> VkResult<impl GPUCommandFuture<Output = RenderRes<Box<dyn BufferLike>>>> {
        let entry_layout_all = self
            .layout
            .one_entry
            .repeat(self.entries.capacity())
            .unwrap()
            .0;
        let expected_buffer_size = entry_layout_all.pad_to_align().size() as u64;
        let needs_copy = matches!(
            self.device().physical_device().memory_model(),
            PhysicalDeviceMemoryModel::Discrete
        );

        // Only non-empty when Selective Updating with `needs_copy == true`
        let mut changes: Vec<vk::BufferCopy> = Vec::new();
        let frame = self
            .frames
            .use_state(|| {
                let buffer = self
                    .allocator
                    .create_dynamic_buffer_uninit(
                        expected_buffer_size,
                        vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR,
                    )
                    .unwrap();
                let contents = buffer.contents_mut().unwrap();
                for i in 0..self.entries.len() {
                    Self::write_sbt_entry(
                        &self.layout,
                        &self.shader_group_handles,
                        &self.entries,
                        self.total_raytype,
                        contents,
                        i,
                    );
                }
                self.changeset
                    .insert(buffer.raw_buffer(), Default::default());
                buffer
            })
            .reuse(|old_buffer| {
                if old_buffer.size() == expected_buffer_size {
                    // Selective updates
                    let write_target = old_buffer.contents_mut().unwrap();
                    let changeset =
                        std::mem::take(self.changeset.get_mut(&old_buffer.raw_buffer()).unwrap());
                    for changed_location in changeset.iter() {
                        Self::write_sbt_entry(
                            &self.layout,
                            &self.shader_group_handles,
                            &self.entries,
                            self.total_raytype,
                            write_target,
                            *changed_location,
                        );
                    }
                    if needs_copy {
                        changes.extend(changeset.into_iter().merge_ranges().map(
                            |(start, size)| {
                                let offset = self
                                    .layout
                                    .one_entry
                                    .repeat(start)
                                    .unwrap()
                                    .0
                                    .pad_to_align()
                                    .size() as u64;
                                let size =
                                    self.layout.one_entry.repeat(size).unwrap().0.size() as u64;
                                vk::BufferCopy {
                                    src_offset: offset,
                                    dst_offset: offset,
                                    size,
                                }
                            },
                        ));
                    }
                } else {
                    // Need to recreate the buffer
                    let buffer = self
                        .allocator
                        .create_dynamic_buffer_uninit(
                            expected_buffer_size,
                            vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR,
                        )
                        .unwrap();

                    self.changeset.remove(&old_buffer.raw_buffer());
                    self.changeset
                        .insert(buffer.raw_buffer(), Default::default());
                    let contents = buffer.contents_mut().unwrap();
                    for i in 0..self.entries.len() {
                        Self::write_sbt_entry(
                            &self.layout,
                            &self.shader_group_handles,
                            &self.entries,
                            self.total_raytype,
                            contents,
                            i,
                        );
                    }
                    *old_buffer = buffer
                }
            });
        let device_buffer = if needs_copy {
            let device_buffer = use_shared_state(
                &mut self.device_buffer,
                |_| {
                    let buffer = self
                        .allocator
                        .create_device_buffer_uninit(
                            expected_buffer_size,
                            vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR,
                        )
                        .unwrap();
                    buffer
                },
                |a| expected_buffer_size != a.size(),
            );
            Some(device_buffer)
        } else {
            None
        };
        let future = commands! {
            let updated_buffer = RenderRes::new(frame);
            if let Some(device_buffer) = device_buffer {
                let mut device_buffer = RenderRes::new(device_buffer);
                if changes.is_empty() {
                    copy_buffer(&updated_buffer, &mut device_buffer)
                } else {
                    copy_buffer_regions(&updated_buffer, &mut device_buffer, changes)
                }.await;
                device_buffer.map(|a| Box::new(a) as Box<dyn BufferLike>)
            } else {
                updated_buffer.map(|a| Box::new(a) as Box<dyn BufferLike>)
            }
        };
        Ok(future)
    }

    fn write_sbt_entry(
        layout: &SbtLayout,
        shader_group_handles: &SbtHandles,
        entries: &[T],
        total_raytype: u32,
        write_target: &mut [u8],
        location: usize,
    ) {
        let size = layout.one_entry.pad_to_align().size();
        let entry_write_target = &mut write_target[size * location..size * (location + 1)];

        let entry = &entries[location];
        // For each raytype
        for i in 0..total_raytype {
            let size = layout.one_raytype.pad_to_align().size();
            let raytype_write_target =
                &mut entry_write_target[size * i as usize..size * (i as usize + 1)];

            let hitgroup_index = entry.hitgroup_index(i);
            let shader_data = shader_group_handles.hitgroup(hitgroup_index);
            raytype_write_target[..shader_data.len()].copy_from_slice(shader_data);

            let parameters = entry.parameter(i);
            let parameters_slice = unsafe {
                std::slice::from_raw_parts(
                    &parameters as *const _ as *const u8,
                    std::mem::size_of_val(&parameters),
                )
            };
            raytype_write_target[layout.handle_size..layout.handle_size + parameters_slice.len()]
                .copy_from_slice(parameters_slice);
        }
    }
}

struct SbtLayout {
    /// The layout for one raytype.
    /// | Raytype 1                                    |
    /// | shader_handles | inline_parameters | padding |
    /// | <--              size           -> | align   |
    one_raytype: Layout,

    // The layout for one entry with all its raytypes
    /// | Raytype 1                                    | Raytype 2                                    |
    /// | shader_handles | inline_parameters | padding | shader_handles | inline_parameters | padding |
    /// | <---                                      size                               ---> |  align  |
    one_entry: Layout,

    /// The size of the shader group handles, padded.
    /// | Raytype 1                                    |
    /// | shader_handles | inline_parameters | padding |
    /// | <--- size ---> |
    handle_size: usize,
}

impl SbtLayout {
    pub fn new<T: HitgroupSbtEntry>(
        shader_group_handles: &SbtHandles,
        total_raytypes: u32,
    ) -> Self {
        let one_raytype = shader_group_handles
            .handle_layout()
            .extend(Layout::new::<T::ShaderParameter>())
            .unwrap()
            .0;

        let one_entry = one_raytype.repeat(total_raytypes as usize).unwrap().0;

        let handle_size = shader_group_handles.handle_layout().pad_to_align().size();
        Self {
            one_raytype,
            one_entry,
            handle_size,
        }
    }
}
