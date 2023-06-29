use crate::{
    commands::{CommandBufferLike, SharedCommandPool},
    BufferLike, HasDevice, ImageLike, QueueRef,
};
use ash::vk;
use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
    fmt::Debug,
    pin::Pin,
    task::Poll,
};

use super::{Disposable, Dispose, GPUCommandFuture};

#[derive(Debug)]
pub struct ResTrackingInfo {
    pub prev_stage_access: Access,
    pub current_stage_access: Access,
    pub last_accessed_stage_index: u32,

    pub queue_family: u32,
    pub queue_index: QueueRef,
    pub prev_queue_family: u32,
    pub prev_queue_index: QueueRef,
    pub last_accessed_timeline: u32,

    pub untracked_semaphore: Option<vk::Semaphore>,
}
impl Default for ResTrackingInfo {
    fn default() -> Self {
        Self {
            prev_stage_access: Access::default(),
            current_stage_access: Access::default(),
            last_accessed_stage_index: 0,

            queue_family: vk::QUEUE_FAMILY_IGNORED,
            queue_index: QueueRef::null(),
            prev_queue_index: QueueRef::null(),
            prev_queue_family: vk::QUEUE_FAMILY_IGNORED,
            last_accessed_timeline: 0,

            untracked_semaphore: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TrackingFeedback {
    pub queue_family: u32,
    pub queue_index: QueueRef,
    pub access: Access,
    pub layout: vk::ImageLayout,
    pub reused: bool,
}
impl Default for TrackingFeedback {
    fn default() -> Self {
        Self {
            queue_family: vk::QUEUE_FAMILY_IGNORED,
            queue_index: QueueRef::null(),
            access: Access::default(),
            layout: vk::ImageLayout::UNDEFINED,
            reused: false,
        }
    }
}
/// A marker trait for things to be placed inside `RenderRes` or `RenderImage`.
/// Applications need to implement this for all types that they want to use inside
/// `RenderRes` or `RenderImage`. Once Rust specialization is implemented and major
/// bugs addressed, we can add a blanket implementation for all types.
/// `impl<T> RenderData for T {}``
pub trait RenderData {
    // This method is called at the end of each objects lifetime when we've verified that they have completed
    // execution on the GPU.
    fn tracking_feedback(&mut self, _feedback: &TrackingFeedback) {}
}
impl RenderData for () {}
impl RenderData for vk::Buffer {}
impl RenderData for vk::Image {}
macro_rules! impl_tuple {
    ($($idx:tt $t:tt),+) => {
        impl<$($t,)+> RenderData for ($($t,)+)
        where
            $($t: RenderData,)+
        {
            fn tracking_feedback(&mut self, feedback: &TrackingFeedback) {
                $(
                    $t :: tracking_feedback(&mut self.$idx, feedback);
                )+
            }
        }
    };
}

impl_tuple!(0 A);
impl_tuple!(0 A, 1 B);
impl_tuple!(0 A, 1 B, 2 C);
impl_tuple!(0 A, 1 B, 2 C, 3 D);
impl_tuple!(0 A, 1 B, 2 C, 3 D, 4 E);
impl_tuple!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F);
impl_tuple!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G);
impl_tuple!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G, 7 H);

pub struct RenderRes<T: RenderData> {
    pub tracking_info: RefCell<ResTrackingInfo>,
    pub inner: T,
    dispose_marker: Dispose<T>,
}

impl<T: RenderData + Debug> Debug for RenderRes<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = std::any::type_name::<Self>();
        let mut f = f.debug_struct(name);

        f.field("inner", &self.inner);

        let tracking = self.tracking_info.borrow();
        f.field(
            "meta",
            &format_args!(
                "last accessed timeline# {}, stage# {}",
                tracking.last_accessed_timeline, tracking.last_accessed_stage_index
            ),
        )
        .field(
            "access",
            &format_args!(
                "{:?} -> {:?}",
                tracking.prev_stage_access, tracking.current_stage_access
            ),
        )
        .field(
            "queue_family",
            &format_args!(
                "{:?} -> {:?}",
                tracking.prev_queue_family, tracking.queue_family
            ),
        )
        .field(
            "queue_index",
            &format_args!(
                "{:?} -> {:?}",
                tracking.prev_queue_index, tracking.queue_index
            ),
        )
        .field("untracked_semaphore", &tracking.untracked_semaphore)
        .finish_non_exhaustive()
    }
}
impl<T: RenderData> Disposable for RenderRes<T> {
    fn retire(&mut self) {
        let tracking = self.tracking_info.borrow();
        self.inner.tracking_feedback(&TrackingFeedback {
            queue_family: tracking.queue_family,
            queue_index: tracking.queue_index,
            access: tracking.current_stage_access.clone(),
            layout: vk::ImageLayout::UNDEFINED,
            reused: true,
        });
    }
    fn dispose(self) {
        self.dispose_marker.dispose();
    }
}
impl<T: RenderData> RenderRes<T> {
    pub fn new(inner: T) -> Self {
        Self {
            tracking_info: Default::default(),
            inner,
            dispose_marker: Dispose::new(),
        }
    }
    pub fn touched(&self) -> bool {
        let tracking = self.tracking_info.borrow();
        tracking.last_accessed_timeline != 0 || tracking.last_accessed_stage_index != 0
    }
    pub(crate) fn with_feedback(inner: T, feedback: &TrackingFeedback) -> Self {
        let this = Self::new(inner);
        {
            let mut tracking = this.tracking_info.borrow_mut();
            tracking.current_stage_access = feedback.access.clone();
            tracking.queue_family = feedback.queue_family;
            tracking.queue_index = feedback.queue_index;
        }
        this
    }
    pub fn inner(&self) -> &T {
        &self.inner
    }
    pub fn inner_mut(&mut self) -> &mut T {
        &mut self.inner
    }
    pub fn into_inner(self) -> T {
        self.dispose_marker.dispose();
        self.inner
    }
    pub fn inspect(mut self, mapper: impl FnOnce(&mut T)) -> Self {
        mapper(&mut self.inner);
        self
    }

    pub fn map<RET: RenderData>(self, mapper: impl FnOnce(T) -> RET) -> RenderRes<RET> {
        self.dispose_marker.dispose();
        RenderRes {
            inner: (mapper)(self.inner),
            tracking_info: self.tracking_info,
            dispose_marker: Dispose::new(),
        }
    }
    pub fn take(self) -> (RenderRes<()>, T) {
        self.dispose_marker.dispose();
        let item = self.inner;
        let res = RenderRes {
            inner: (),
            tracking_info: self.tracking_info,
            dispose_marker: Dispose::new(),
        };
        (res, item)
    }
    pub fn merge<O: RenderData>(self, other: RenderRes<O>) -> RenderRes<(T, O)> {
        let self_tracking = self.tracking_info.borrow();
        let other_tracking = other.tracking_info.borrow();

        let queue_family = if self_tracking.queue_family == vk::QUEUE_FAMILY_IGNORED {
            other_tracking.queue_family
        } else {
            self_tracking.queue_family
        };
        let prev_queue_family = if other_tracking.queue_family == vk::QUEUE_FAMILY_IGNORED {
            self_tracking.queue_family
        } else {
            other_tracking.queue_family
        };
        assert!(
            self_tracking.queue_family == vk::QUEUE_FAMILY_IGNORED
                || other_tracking.queue_family == vk::QUEUE_FAMILY_IGNORED
                || self_tracking.queue_family == other_tracking.queue_family
        );
        assert!(
            self_tracking.prev_queue_family == vk::QUEUE_FAMILY_IGNORED
                || other_tracking.prev_queue_family == vk::QUEUE_FAMILY_IGNORED
                || self_tracking.prev_queue_family == other_tracking.prev_queue_family
        );
        assert_eq!(
            self_tracking.prev_queue_family,
            other_tracking.prev_queue_family
        );
        assert!(
            self_tracking.untracked_semaphore.is_none()
                || other_tracking.untracked_semaphore.is_none()
        );
        let merged_tracking = ResTrackingInfo {
            prev_stage_access: self_tracking
                .prev_stage_access
                .merge(&other_tracking.prev_stage_access),
            current_stage_access: self_tracking
                .current_stage_access
                .merge(&other_tracking.current_stage_access),
            last_accessed_stage_index: self_tracking
                .last_accessed_stage_index
                .max(other_tracking.last_accessed_stage_index),
            queue_family,
            queue_index: self_tracking.queue_index.max(other_tracking.queue_index),
            prev_queue_family,
            prev_queue_index: self_tracking
                .prev_queue_index
                .max(other_tracking.prev_queue_index),
            last_accessed_timeline: self_tracking
                .last_accessed_timeline
                .max(other_tracking.last_accessed_timeline),
            untracked_semaphore: self_tracking
                .untracked_semaphore
                .or(other_tracking.untracked_semaphore),
        };
        self.dispose_marker.dispose();
        other.dispose_marker.dispose();
        RenderRes {
            tracking_info: RefCell::new(merged_tracking),
            inner: (self.inner, other.inner),
            dispose_marker: Dispose::new(),
        }
    }
}

pub struct RenderImage<T: RenderData> {
    pub res: RenderRes<T>,
    pub old_layout: Cell<vk::ImageLayout>,
    pub layout: Cell<vk::ImageLayout>,
}
impl<T: RenderData + Debug> Debug for RenderImage<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = std::any::type_name::<Self>();
        let mut f = f.debug_struct(name);

        f.field("inner", &self.res.inner);

        let tracking = self.res.tracking_info.borrow();
        f.field(
            "meta",
            &format_args!(
                "last accessed timeline# {}, stage# {}",
                tracking.last_accessed_timeline, tracking.last_accessed_stage_index
            ),
        )
        .field(
            "access",
            &format_args!(
                "{:?} -> {:?}",
                tracking.prev_stage_access, tracking.current_stage_access
            ),
        )
        .field(
            "queue_family",
            &format_args!(
                "{:?} -> {:?}",
                tracking.prev_queue_family, tracking.queue_family
            ),
        )
        .field(
            "queue_index",
            &format_args!(
                "{:?} -> {:?}",
                tracking.prev_queue_index, tracking.queue_index
            ),
        )
        .field("untracked_semaphore", &tracking.untracked_semaphore)
        .field(
            "layout",
            &format_args!("{:?} -> {:?}", self.old_layout.get(), self.layout.get()),
        )
        .finish_non_exhaustive()
    }
}
impl<T: RenderData> Disposable for RenderImage<T> {
    fn retire(&mut self) {
        let tracking = self.res.tracking_info.borrow();
        self.res.inner.tracking_feedback(&TrackingFeedback {
            queue_family: tracking.queue_family,
            queue_index: tracking.queue_index,
            access: tracking.current_stage_access.clone(),
            layout: self.layout.get(),
            reused: true,
        });
    }
    fn dispose(self) {
        self.res.dispose()
    }
}
impl<T: RenderData> RenderImage<T> {
    pub fn new(inner: T, initial_layout: vk::ImageLayout) -> Self {
        Self {
            res: RenderRes::new(inner),
            layout: Cell::new(initial_layout),
            old_layout: Cell::new(initial_layout),
        }
    }
    pub fn touched(&self) -> bool {
        self.res.touched()
    }
    pub(crate) fn with_feedback(inner: T, feedback: &TrackingFeedback) -> Self {
        let this = Self::new(inner, feedback.layout);
        {
            let mut tracking = this.res.tracking_info.borrow_mut();
            tracking.current_stage_access = feedback.access.clone();
            tracking.queue_family = feedback.queue_family;
            tracking.queue_index = feedback.queue_index;
        }
        this
    }
    pub fn inner(&self) -> &T {
        &self.res.inner
    }
    pub fn inner_mut(&mut self) -> &mut T {
        &mut self.res.inner
    }
    pub fn into_inner(self) -> T {
        self.res.into_inner()
    }
    pub fn map<RET: RenderData>(self, mapper: impl FnOnce(T) -> RET) -> RenderImage<RET> {
        let res = self.res.map(mapper);
        RenderImage {
            res,
            old_layout: self.old_layout,
            layout: self.layout,
        }
    }
}

/// One per command buffer record call. If multiple command buffers were merged together on the queue level,
/// this would be the same.
pub struct CommandBufferRecordContext<'a> {
    // perhaps also a reference to the command buffer allocator
    pub stage_index: u32,
    pub timeline_index: u32,
    pub queue: QueueRef,
    pub command_buffers: &'a mut Vec<vk::CommandBuffer>,
    pub recording_command_buffer: &'a mut Option<vk::CommandBuffer>,
    pub command_pool: &'a mut SharedCommandPool,
}
impl<'a> HasDevice for CommandBufferRecordContext<'a> {
    fn device(&self) -> &std::sync::Arc<crate::Device> {
        self.command_pool.device()
    }
}
impl<'a> CommandBufferRecordContext<'a> {
    pub fn queue_family_index(&self) -> u32 {
        self.command_pool.queue_family_index()
    }
    /// Immediatly record a command buffer, allocated from the shared command pool.
    pub fn record(&mut self, callback: impl FnOnce(&Self, vk::CommandBuffer)) {
        let command_buffer = if let Some(command_buffer) = self.recording_command_buffer.take() {
            command_buffer
        } else {
            let buffer = self.command_pool.allocate_one();
            unsafe {
                self.device()
                    .begin_command_buffer(
                        buffer,
                        &vk::CommandBufferBeginInfo {
                            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                            ..Default::default()
                        },
                    )
                    .unwrap();
            }
            buffer
        };
        callback(self, command_buffer);
        *self.recording_command_buffer = Some(command_buffer);
    }
    pub fn add_command_buffer<T: CommandBufferLike + 'a>(&mut self, buffer: T) {
        if let Some(command_buffer) = self.recording_command_buffer.take() {
            unsafe {
                self.device().end_command_buffer(command_buffer).unwrap();
            }
            self.command_buffers.push(command_buffer);
        }
        self.command_buffers.push(buffer.raw_command_buffer());
    }
}

impl<'host> CommandBufferRecordContext<'host> {
    pub fn current_stage_index(&self) -> u32 {
        self.stage_index
    }
}
#[derive(Clone, Debug)]
pub(crate) struct StageImageBarrier {
    pub barrier: vk::MemoryBarrier2,
    pub src_layout: vk::ImageLayout,
    pub dst_layout: vk::ImageLayout,
    pub src_queue_family: u32,
    pub dst_queue_family: u32,
    pub src_queue: QueueRef,
    pub dst_queue: QueueRef,
}
#[derive(Clone)]
pub(crate) struct StageContextImage {
    pub image: vk::Image,
    pub subresource_range: vk::ImageSubresourceRange,
    pub extent: vk::Extent3D,
}

impl PartialEq for StageContextImage {
    fn eq(&self, other: &Self) -> bool {
        self.image == other.image
            && self.subresource_range.aspect_mask == other.subresource_range.aspect_mask
            && self.subresource_range.base_array_layer == other.subresource_range.base_array_layer
            && self.subresource_range.base_mip_level == other.subresource_range.base_mip_level
            && self.subresource_range.level_count == other.subresource_range.level_count
            && self.subresource_range.layer_count == other.subresource_range.layer_count
    }
}
impl Eq for StageContextImage {}

impl PartialOrd for StageContextImage {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for StageContextImage {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.image.cmp(&other.image) {
            core::cmp::Ordering::Equal => {}
            ord => return ord,
        }
        match self
            .subresource_range
            .base_array_layer
            .cmp(&other.subresource_range.base_array_layer)
        {
            core::cmp::Ordering::Equal => {}
            ord => return ord,
        }
        match self
            .subresource_range
            .layer_count
            .cmp(&other.subresource_range.layer_count)
        {
            core::cmp::Ordering::Equal => {}
            ord => return ord,
        }
        match self
            .subresource_range
            .base_mip_level
            .cmp(&other.subresource_range.base_mip_level)
        {
            core::cmp::Ordering::Equal => {}
            ord => return ord,
        }
        match self
            .subresource_range
            .level_count
            .cmp(&other.subresource_range.level_count)
        {
            core::cmp::Ordering::Equal => {}
            ord => return ord,
        }
        self.subresource_range
            .aspect_mask
            .cmp(&other.subresource_range.aspect_mask)
    }
}

#[derive(Clone)]
pub(crate) struct StageBufferBarrier {
    pub barrier: vk::MemoryBarrier2,
    pub src_queue_family: u32,
    pub dst_queue_family: u32,
    pub src_queue: QueueRef,
    pub dst_queue: QueueRef,
}
#[derive(Clone)]
pub(crate) struct StageContextBuffer {
    pub buffer: vk::Buffer,
    pub offset: vk::DeviceSize,
    pub size: vk::DeviceSize,
}
impl BufferLike for StageContextBuffer {
    fn raw_buffer(&self) -> vk::Buffer {
        self.buffer
    }

    fn size(&self) -> vk::DeviceSize {
        self.size
    }
    fn offset(&self) -> vk::DeviceSize {
        self.offset
    }
    fn device_address(&self) -> vk::DeviceAddress {
        unimplemented!()
    }
    fn as_mut_ptr(&mut self) -> Option<*mut u8> {
        unimplemented!()
    }
}
impl PartialEq for StageContextBuffer {
    fn eq(&self, other: &Self) -> bool {
        self.buffer == other.buffer && self.offset == other.offset && self.size == other.size
    }
}
impl Eq for StageContextBuffer {}
impl PartialOrd for StageContextBuffer {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for StageContextBuffer {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.offset.cmp(&other.offset) {
            core::cmp::Ordering::Equal => {}
            ord => return ord,
        }
        match self.size.cmp(&other.size) {
            core::cmp::Ordering::Equal => {}
            ord => return ord,
        }
        self.buffer.cmp(&other.buffer)
    }
}

pub enum StageContextSemaphoreTransition {
    Managed {
        src_queue: QueueRef,
        dst_queue: QueueRef,
        src_stages: vk::PipelineStageFlags2,
        dst_stages: vk::PipelineStageFlags2,
    },
    Untracked {
        semaphore: vk::Semaphore,
        dst_queue: QueueRef,
        dst_stages: vk::PipelineStageFlags2,
    },
}

pub struct StageContext {
    pub stage_index: u32,
    pub timeline_index: u32,
    pub queue_family_index: u32,
    pub queue_index: QueueRef,
    pub global_access: vk::MemoryBarrier2,
    pub(crate) image_accesses: BTreeMap<StageContextImage, StageImageBarrier>,
    pub(crate) buffer_accesses: BTreeMap<StageContextBuffer, StageBufferBarrier>,

    // Queue, srcQueue, dstQueue, srcStages, dstStages
    pub semaphore_transitions: Vec<StageContextSemaphoreTransition>,
}

impl StageContext {
    pub fn new(
        stage_index: u32,
        timeline_index: u32,
        queue_family_index: u32,
        queue_index: QueueRef,
    ) -> Self {
        Self {
            stage_index,
            queue_family_index,
            queue_index,
            timeline_index,
            global_access: vk::MemoryBarrier2::default(),
            image_accesses: BTreeMap::new(),
            buffer_accesses: BTreeMap::new(),
            semaphore_transitions: Vec::new(),
        }
    }
    fn add_barrier_tracking(&mut self, tracking: &mut ResTrackingInfo, access: &Access) {
        if tracking.last_accessed_timeline < self.timeline_index {
            let last_accessed_timeline = tracking.last_accessed_timeline;
            tracking.prev_queue_family =
                std::mem::replace(&mut tracking.queue_family, self.queue_family_index);
            tracking.prev_queue_index =
                std::mem::replace(&mut tracking.queue_index, self.queue_index);
            // Need semaphore sync
            // queue, timeline: signal at pipeline barrier.
            // If an earlier stage was already signaled we need to make another signal.
            // If a later stage was already signaled, we can
            // ok this is very problematic.

            // In the binary semaphore model, each semaphore can only be waited on once.
            // This is great for our purpose.
            // We can say unconditionally: This queue, signal on this stage. (what is "this stage?")
            tracking.last_accessed_timeline = self.timeline_index;

            if let Some(untracked_semaphore) = tracking.untracked_semaphore.as_mut() {
                if *untracked_semaphore == vk::Semaphore::null() {
                    panic!("Attempts to wait on vk::AcquireNextImageKHR twice.");
                }

                let mut barrier = vk::MemoryBarrier2::default();
                // TODO: Do we have to consider image layout transfers here?
                get_memory_access(&mut barrier, &tracking.prev_stage_access, access, false);

                self.semaphore_transitions
                    .push(StageContextSemaphoreTransition::Untracked {
                        semaphore: *untracked_semaphore,
                        dst_queue: tracking.queue_index,
                        dst_stages: barrier.dst_stage_mask,
                    });

                *untracked_semaphore = vk::Semaphore::null();
            } else if last_accessed_timeline != 0 {
                assert!(!tracking.prev_queue_index.is_null());
                assert!(tracking.prev_queue_family != vk::QUEUE_FAMILY_IGNORED);
                let mut barrier = vk::MemoryBarrier2::default();
                // TODO: Do we have to consider image layout transfers here?
                get_memory_access(&mut barrier, &tracking.prev_stage_access, access, false);
                self.semaphore_transitions
                    .push(StageContextSemaphoreTransition::Managed {
                        src_queue: tracking.prev_queue_index,
                        dst_queue: tracking.queue_index,
                        src_stages: barrier.src_stage_mask,
                        dst_stages: barrier.dst_stage_mask,
                    });
            }
        }
    }
    /// Declare a global memory write
    #[inline]
    pub fn write<T: RenderData>(
        &mut self,
        res: &mut RenderRes<T>,
        stages: vk::PipelineStageFlags2,
        accesses: vk::AccessFlags2,
    ) {
        assert!(access_flag_is_write(accesses), "Expected write accesses");
        let access = Access {
            write_access: accesses,
            write_stages: stages,
            ..Default::default()
        };

        let mut tracking = res.tracking_info.borrow_mut();
        if tracking.last_accessed_stage_index < self.stage_index
            || tracking.last_accessed_timeline < self.timeline_index
        {
            tracking.prev_stage_access = std::mem::take(&mut tracking.current_stage_access);
        }
        self.add_barrier_tracking(&mut tracking, &access);
        // Writes never need to worry about queue family ownership transfers.
        get_memory_access(
            &mut self.global_access,
            &tracking.prev_stage_access,
            &access,
            false,
        );
        tracking.current_stage_access.write_access |= accesses;
        tracking.current_stage_access.write_stages |= stages;
        tracking.last_accessed_stage_index = self.stage_index;
    }
    /// Declare a global memory read
    #[inline]
    pub fn read<T: RenderData>(
        &mut self,
        res: &RenderRes<T>,
        stages: vk::PipelineStageFlags2,
        accesses: vk::AccessFlags2,
    ) where
        T: BufferLike,
    {
        assert!(access_flag_is_read(accesses), "Expected read accesses");
        let access = Access {
            read_access: accesses,
            read_stages: stages,
            ..Default::default()
        };
        let mut tracking = &mut *res.tracking_info.borrow_mut();
        if tracking.last_accessed_stage_index < self.stage_index
            || tracking.last_accessed_timeline < self.timeline_index
        {
            tracking.prev_stage_access = std::mem::take(&mut tracking.current_stage_access);

            // Read after write does not need memory barriers. Therefore old memory access info must be carried forward.
            tracking.current_stage_access.write_access |= tracking.prev_stage_access.write_access;
            tracking.current_stage_access.write_stages |= tracking.prev_stage_access.write_stages;
        }
        self.add_barrier_tracking(&mut tracking, &access);
        if tracking.prev_queue_family == self.queue_family_index
            || tracking.prev_queue_family == vk::QUEUE_FAMILY_IGNORED
        {
            get_memory_access(
                &mut self.global_access,
                &tracking.prev_stage_access,
                &access,
                false,
            );
        } else {
            let buffer_barrier = self
                .buffer_accesses
                .entry(StageContextBuffer {
                    buffer: res.inner().raw_buffer(),
                    offset: res.inner().offset(),
                    size: res.inner().size(),
                })
                .or_insert(StageBufferBarrier {
                    barrier: Default::default(),
                    src_queue_family: tracking.prev_queue_family,
                    dst_queue_family: self.queue_family_index,
                    src_queue: tracking.prev_queue_index,
                    dst_queue: tracking.queue_index,
                });
            get_memory_access(
                &mut buffer_barrier.barrier,
                &tracking.prev_stage_access,
                &access,
                false,
            );
        }
        tracking.current_stage_access.read_access |= accesses;
        tracking.current_stage_access.read_stages |= stages;
        tracking.last_accessed_stage_index = self.stage_index;
    }
    /// Declare a global memory read
    #[inline]
    pub fn read_others<T: RenderData>(
        &mut self,
        res: &RenderRes<T>,
        stages: vk::PipelineStageFlags2,
        accesses: vk::AccessFlags2,
    ) {
        assert!(access_flag_is_read(accesses), "Expected read accesses");
        let access = Access {
            read_access: accesses,
            read_stages: stages,
            ..Default::default()
        };
        let mut tracking = &mut *res.tracking_info.borrow_mut();
        if tracking.last_accessed_stage_index < self.stage_index
            || tracking.last_accessed_timeline < self.timeline_index
        {
            tracking.prev_stage_access = std::mem::take(&mut tracking.current_stage_access);

            // Read after write does not need memory barriers. Therefore old memory access info must be carried forward.
            tracking.current_stage_access.write_access |= tracking.prev_stage_access.write_access;
            tracking.current_stage_access.write_stages |= tracking.prev_stage_access.write_stages;
        }
        self.add_barrier_tracking(&mut tracking, &access);
        if tracking.prev_queue_family == self.queue_family_index
            || tracking.prev_queue_family == vk::QUEUE_FAMILY_IGNORED
        {
            get_memory_access(
                &mut self.global_access,
                &tracking.prev_stage_access,
                &access,
                false,
            );
        } else {
            unimplemented!()
        }
        tracking.current_stage_access.read_access |= accesses;
        tracking.current_stage_access.read_stages |= stages;
        tracking.last_accessed_stage_index = self.stage_index;
    }
    #[inline]
    pub fn write_image<T: RenderData>(
        &mut self,
        res: &mut RenderImage<T>,
        stages: vk::PipelineStageFlags2,
        accesses: vk::AccessFlags2,
        layout: vk::ImageLayout,
    ) where
        T: ImageLike,
    {
        assert!(access_flag_is_write(accesses), "Expected write accesses");
        let access = Access {
            write_access: accesses,
            write_stages: stages,
            ..Default::default()
        };

        let mut tracking = res.res.tracking_info.borrow_mut();
        if tracking.last_accessed_stage_index < self.stage_index
            || tracking.last_accessed_timeline < self.timeline_index
        {
            tracking.prev_stage_access = std::mem::take(&mut tracking.current_stage_access);
            *res.old_layout.get_mut() = std::mem::replace(res.layout.get_mut(), layout);
        } else {
            assert_eq!(
                tracking.queue_family, self.queue_family_index,
                "Layout mismatch."
            );
            assert_eq!(res.layout.get(), layout, "Layout mismatch.");
        }
        self.add_barrier_tracking(&mut tracking, &access);

        if res.layout == res.old_layout
            && (tracking.queue_family == tracking.prev_queue_family
                || tracking.prev_queue_family == vk::QUEUE_FAMILY_IGNORED)
        {
            // Global memory barrier would suffice.
            get_memory_access(
                &mut self.global_access,
                &tracking.prev_stage_access,
                &access,
                false,
            );
        } else {
            let image_barrier = self
                .image_accesses
                .entry(StageContextImage {
                    image: res.inner().raw_image(),
                    subresource_range: res.inner().subresource_range(),
                    extent: res.inner().extent(),
                })
                .or_insert(StageImageBarrier {
                    barrier: Default::default(),
                    src_layout: vk::ImageLayout::UNDEFINED,
                    dst_layout: layout,
                    src_queue_family: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family: vk::QUEUE_FAMILY_IGNORED,
                    src_queue: QueueRef::null(),
                    dst_queue: QueueRef::null(),
                });
            get_memory_access(
                &mut image_barrier.barrier,
                &tracking.prev_stage_access,
                &access,
                image_barrier.dst_layout != image_barrier.src_layout,
            );
        }

        tracking.current_stage_access.write_access |= accesses;
        tracking.current_stage_access.write_stages |= stages;
        tracking.last_accessed_stage_index = self.stage_index;
    }
    /// Declare a global memory read
    #[inline]
    pub fn read_image<T: RenderData>(
        &mut self,
        res: &RenderImage<T>,
        stages: vk::PipelineStageFlags2,
        accesses: vk::AccessFlags2,
        layout: vk::ImageLayout,
    ) where
        T: ImageLike,
    {
        assert!(access_flag_is_read(accesses), "Expected read accesses");
        let access = Access {
            read_access: accesses,
            read_stages: stages,
            ..Default::default()
        };
        let mut tracking = &mut *res.res.tracking_info.borrow_mut();
        if tracking.last_accessed_stage_index < self.stage_index
            || tracking.last_accessed_timeline < self.timeline_index
        {
            tracking.prev_stage_access = std::mem::take(&mut tracking.current_stage_access);
            res.old_layout.replace(res.layout.get());
            res.layout.replace(layout);

            // Read after write does not need memory barriers. Therefore old memory access info must be carried forward.
            // TODO: However, it's unclear how image layout transfer would impact this. So let's just leave it here for now.
        } else {
            assert_eq!(
                tracking.queue_family, self.queue_family_index,
                "Queue family mismatch."
            );
            assert_eq!(res.layout.get(), layout, "Layout mismatch.");
        }
        self.add_barrier_tracking(&mut tracking, &access);

        if res.layout == res.old_layout
            && (tracking.queue_family == tracking.prev_queue_family
                || tracking.prev_queue_family == vk::QUEUE_FAMILY_IGNORED)
        {
            // Global memory barrier would suffice.
            get_memory_access(
                &mut self.global_access,
                &tracking.prev_stage_access,
                &access,
                false,
            );
        } else {
            let image_barrier = self
                .image_accesses
                .entry(StageContextImage {
                    image: res.inner().raw_image(),
                    subresource_range: res.inner().subresource_range(),
                    extent: res.inner().extent(),
                })
                .or_insert(StageImageBarrier {
                    barrier: Default::default(),
                    src_layout: res.old_layout.get(),
                    dst_layout: layout,
                    src_queue_family: tracking.prev_queue_family,
                    dst_queue_family: if tracking.prev_queue_family == vk::QUEUE_FAMILY_IGNORED {
                        vk::QUEUE_FAMILY_IGNORED
                    } else {
                        self.queue_family_index
                    },
                    src_queue: QueueRef::null(),
                    dst_queue: QueueRef::null(),
                });
            get_memory_access(
                &mut image_barrier.barrier,
                &tracking.prev_stage_access,
                &access,
                image_barrier.dst_layout != image_barrier.src_layout,
            );
        }

        tracking.current_stage_access.read_access |= accesses;
        tracking.current_stage_access.read_stages |= stages;
        tracking.last_accessed_stage_index = self.stage_index;
    }
}

impl<'a> CommandBufferRecordContext<'a> {
    pub fn record_one_step<T: GPUCommandFuture>(
        &mut self,
        mut fut: Pin<&'a mut T>,
        recycled_state: &mut T::RecycledState,
        context_handler: impl FnOnce(&StageContext),
    ) -> Poll<(T::Output, T::RetainedState)> {
        let mut next_stage = StageContext::new(
            self.stage_index,
            self.timeline_index,
            self.command_pool.queue_family_index(),
            self.queue,
        );
        fut.as_mut().context(&mut next_stage);
        (context_handler)(&next_stage);

        let _queue = self.queue;
        Self::add_barrier(&next_stage, |dependency_info| {
            self.record(|ctx, command_buffer| unsafe {
                ctx.device()
                    .cmd_pipeline_barrier2(command_buffer, dependency_info);
            });
        });
        let ret = fut.as_mut().record(self, recycled_state);
        ret
    }
    fn add_barrier(
        next_stage: &StageContext,
        cmd_pipeline_barrier: impl FnOnce(&vk::DependencyInfo),
    ) {
        let mut global_memory_barrier = next_stage.global_access;
        let mut image_barriers: Vec<vk::ImageMemoryBarrier2> = Vec::new();
        let mut buffer_barriers: Vec<vk::BufferMemoryBarrier2> = Vec::new();

        // Set the global memory barrier.

        for (image, image_barrier) in next_stage.image_accesses.iter() {
            let barrier = &image_barrier.barrier;
            if image_barrier.src_layout == image_barrier.dst_layout
                && image_barrier.src_queue_family == image_barrier.dst_queue_family
            {
                global_memory_barrier.dst_access_mask |= barrier.dst_access_mask;
                global_memory_barrier.src_access_mask |= barrier.src_access_mask;
                global_memory_barrier.dst_stage_mask |= barrier.dst_stage_mask;
                global_memory_barrier.src_stage_mask |= barrier.src_stage_mask;
            } else {
                // Needs image layout transfer.
                let o = vk::ImageMemoryBarrier2 {
                    src_access_mask: barrier.src_access_mask,
                    src_stage_mask: barrier.src_stage_mask,
                    dst_access_mask: barrier.dst_access_mask,
                    dst_stage_mask: barrier.dst_stage_mask,
                    image: image.image,
                    subresource_range: image.subresource_range,
                    old_layout: image_barrier.src_layout,
                    new_layout: image_barrier.dst_layout,
                    src_queue_family_index: image_barrier.src_queue_family,
                    dst_queue_family_index: image_barrier.dst_queue_family,
                    ..Default::default()
                };
                image_barriers.push(o);
            }
        }
        for (buffer, buffer_barrier) in next_stage.buffer_accesses.iter() {
            assert_ne!(
                buffer_barrier.src_queue_family,
                buffer_barrier.dst_queue_family
            );
            let o = vk::BufferMemoryBarrier2 {
                src_access_mask: buffer_barrier.barrier.src_access_mask,
                src_stage_mask: buffer_barrier.barrier.src_stage_mask,
                dst_access_mask: buffer_barrier.barrier.dst_access_mask,
                dst_stage_mask: buffer_barrier.barrier.dst_stage_mask,
                buffer: buffer.buffer,
                size: buffer.size,
                offset: buffer.offset,
                src_queue_family_index: buffer_barrier.src_queue_family,
                dst_queue_family_index: buffer_barrier.dst_queue_family,
                ..Default::default()
            };
            buffer_barriers.push(o);
        }
        let mut dep = vk::DependencyInfo {
            dependency_flags: vk::DependencyFlags::BY_REGION, // TODO
            ..Default::default()
        };
        if !global_memory_barrier.dst_access_mask.is_empty()
            || !global_memory_barrier.src_access_mask.is_empty()
            || !global_memory_barrier.dst_stage_mask.is_empty()
            || !global_memory_barrier.src_stage_mask.is_empty()
        {
            dep.memory_barrier_count = 1;
            dep.p_memory_barriers = &global_memory_barrier;
        }
        if !image_barriers.is_empty() {
            dep.image_memory_barrier_count = image_barriers.len() as u32;
            dep.p_image_memory_barriers = image_barriers.as_ptr();
        }

        if !buffer_barriers.is_empty() {
            dep.buffer_memory_barrier_count = buffer_barriers.len() as u32;
            dep.p_buffer_memory_barriers = buffer_barriers.as_ptr();
        }

        if dep.memory_barrier_count > 0
            || dep.buffer_memory_barrier_count > 0
            || dep.image_memory_barrier_count > 0
        {
            cmd_pipeline_barrier(&dep);
        }
    }
}

#[derive(Clone, Default, Debug)]
pub struct Access {
    pub read_stages: vk::PipelineStageFlags2,
    pub read_access: vk::AccessFlags2,
    pub write_stages: vk::PipelineStageFlags2,
    pub write_access: vk::AccessFlags2,
}

impl Access {
    pub fn has_read(&self) -> bool {
        !self.read_stages.is_empty() || !self.read_access.is_empty()
    }
    pub fn has_write(&self) -> bool {
        !self.write_stages.is_empty() || !self.write_access.is_empty()
    }
    pub fn merge(&self, other: &Self) -> Self {
        Self {
            read_access: self.read_access | other.read_access,
            write_access: self.write_access | other.write_access,
            write_stages: self.write_stages | other.write_stages,
            read_stages: self.read_stages | other.read_stages,
        }
    }
}

fn get_memory_access(
    memory_barrier: &mut vk::MemoryBarrier2,
    before_access: &Access,
    after_access: &Access,
    had_image_layout_transfer: bool,
) {
    if had_image_layout_transfer {
        memory_barrier.src_stage_mask |= before_access.write_stages | before_access.read_stages;
        memory_barrier.dst_stage_mask |= after_access.write_stages | after_access.read_stages;
        memory_barrier.src_access_mask |= before_access.write_access | before_access.read_access;
        memory_barrier.dst_access_mask |= after_access.write_access | after_access.read_access;
        return;
    }
    if before_access.has_write() && after_access.has_write() {
        // Write after write
        memory_barrier.src_stage_mask |= before_access.write_stages;
        memory_barrier.dst_stage_mask |= after_access.write_stages;
        memory_barrier.src_access_mask |= before_access.write_access;
        memory_barrier.dst_access_mask |= after_access.write_access;
    }
    if before_access.has_read() && after_access.has_write() {
        // Write after read
        memory_barrier.src_stage_mask |= before_access.read_stages;
        memory_barrier.dst_stage_mask |= after_access.write_stages;
        // No need for memory barrier
    }
    if before_access.has_write() && after_access.has_read() {
        // Read after write
        memory_barrier.src_stage_mask |= before_access.write_stages;
        memory_barrier.dst_stage_mask |= after_access.read_stages;
        memory_barrier.src_access_mask |= before_access.write_access;
        memory_barrier.dst_access_mask |= after_access.read_access;
    }
}

fn access_flag_is_write(flags: vk::AccessFlags2) -> bool {
    let all_write_bits: vk::AccessFlags2 = vk::AccessFlags2::SHADER_WRITE
        | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
        | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
        | vk::AccessFlags2::TRANSFER_WRITE
        | vk::AccessFlags2::HOST_WRITE
        | vk::AccessFlags2::MEMORY_WRITE
        | vk::AccessFlags2::SHADER_STORAGE_WRITE
        | vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR
        | vk::AccessFlags2::VIDEO_ENCODE_WRITE_KHR
        | vk::AccessFlags2::TRANSFORM_FEEDBACK_WRITE_EXT
        | vk::AccessFlags2::TRANSFORM_FEEDBACK_COUNTER_WRITE_EXT
        | vk::AccessFlags2::COMMAND_PREPROCESS_WRITE_NV
        | vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR
        | vk::AccessFlags2::MICROMAP_WRITE_EXT
        | vk::AccessFlags2::OPTICAL_FLOW_WRITE_NV;

    // Clear all the write bits. If nothing is left, that means there's no read bits.
    flags & !all_write_bits == vk::AccessFlags2::NONE
}

fn access_flag_is_read(flags: vk::AccessFlags2) -> bool {
    let all_read_bits: vk::AccessFlags2 = vk::AccessFlags2::INDIRECT_COMMAND_READ
        | vk::AccessFlags2::INDEX_READ
        | vk::AccessFlags2::VERTEX_ATTRIBUTE_READ
        | vk::AccessFlags2::UNIFORM_READ
        | vk::AccessFlags2::INPUT_ATTACHMENT_READ
        | vk::AccessFlags2::SHADER_READ
        | vk::AccessFlags2::COLOR_ATTACHMENT_READ
        | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
        | vk::AccessFlags2::TRANSFER_READ
        | vk::AccessFlags2::HOST_READ
        | vk::AccessFlags2::MEMORY_READ
        | vk::AccessFlags2::SHADER_SAMPLED_READ
        | vk::AccessFlags2::SHADER_STORAGE_READ
        | vk::AccessFlags2::VIDEO_DECODE_READ_KHR
        | vk::AccessFlags2::VIDEO_ENCODE_READ_KHR
        | vk::AccessFlags2::TRANSFORM_FEEDBACK_COUNTER_READ_EXT
        | vk::AccessFlags2::CONDITIONAL_RENDERING_READ_EXT
        | vk::AccessFlags2::COMMAND_PREPROCESS_READ_NV
        | vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR
        | vk::AccessFlags2::FRAGMENT_DENSITY_MAP_READ_EXT
        | vk::AccessFlags2::COLOR_ATTACHMENT_READ_NONCOHERENT_EXT
        | vk::AccessFlags2::DESCRIPTOR_BUFFER_READ_EXT
        | vk::AccessFlags2::INVOCATION_MASK_READ_HUAWEI
        | vk::AccessFlags2::SHADER_BINDING_TABLE_READ_KHR
        | vk::AccessFlags2::MICROMAP_READ_EXT
        | vk::AccessFlags2::OPTICAL_FLOW_READ_NV;

    // Clear all the write bits. If nothing is left, that means there's no read bits.
    flags & !all_read_bits == vk::AccessFlags2::NONE
}
#[cfg(any())]
mod tests {
    use super::*;
    fn assert_global(
        dep: &vk::DependencyInfo,
        src_stage_mask: vk::PipelineStageFlags2,
        src_access_mask: vk::AccessFlags2,
        dst_stage_mask: vk::PipelineStageFlags2,
        dst_access_mask: vk::AccessFlags2,
    ) {
        assert_eq!(dep.memory_barrier_count, 1);
        assert_eq!(dep.buffer_memory_barrier_count, 0);
        assert_eq!(dep.image_memory_barrier_count, 0);
        let memory_barrier = unsafe { &*dep.p_memory_barriers };
        assert_eq!(memory_barrier.src_stage_mask, src_stage_mask);
        assert_eq!(memory_barrier.src_access_mask, src_access_mask);
        assert_eq!(memory_barrier.dst_stage_mask, dst_stage_mask);
        assert_eq!(memory_barrier.dst_access_mask, dst_access_mask);
        assert_ne!(memory_barrier.src_stage_mask, vk::PipelineStageFlags2::NONE);
        assert_ne!(memory_barrier.dst_stage_mask, vk::PipelineStageFlags2::NONE);
    }

    use vk::AccessFlags2 as A;
    use vk::PipelineStageFlags2 as S;
    enum ReadWrite {
        Read(vk::PipelineStageFlags2, vk::AccessFlags2),
        Write(vk::PipelineStageFlags2, vk::AccessFlags2),
        ReadWrite(Access),
    }
    impl ReadWrite {
        fn stage<T: BufferLike>(&self, stage_ctx: &mut StageContext, res: &mut RenderRes<T>) {
            match &self {
                ReadWrite::Read(stage, access) => stage_ctx.read(res, *stage, *access),
                ReadWrite::Write(stage, access) => stage_ctx.write(res, *stage, *access),
                ReadWrite::ReadWrite(access) => {
                    stage_ctx.read(res, access.read_stages, access.read_access);
                    stage_ctx.write(res, access.write_stages, access.write_access);
                }
            }
        }
    }

    fn make_stage(stage_index: u32) -> StageContext {
        StageContext::new(stage_index, 0, vk::QUEUE_FAMILY_IGNORED, QueueRef(0))
    }

    #[test]
    fn c2c_global_tests() {
        let test_cases = [
            (
                [
                    ReadWrite::Read(S::COMPUTE_SHADER, A::SHADER_STORAGE_READ),
                    ReadWrite::Read(S::COMPUTE_SHADER, A::SHADER_READ),
                ],
                (S::NONE, A::NONE, S::NONE, A::NONE),
            ), // RaR
            (
                [
                    ReadWrite::Read(S::COMPUTE_SHADER, A::SHADER_STORAGE_READ),
                    ReadWrite::Write(S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE),
                ],
                (S::COMPUTE_SHADER, A::NONE, S::COMPUTE_SHADER, A::NONE),
            ), // WaR
            (
                [
                    ReadWrite::Write(S::COMPUTE_SHADER, A::SHADER_STORAGE_READ),
                    ReadWrite::Read(S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE),
                ],
                (
                    S::COMPUTE_SHADER,
                    A::SHADER_STORAGE_READ,
                    S::COMPUTE_SHADER,
                    A::SHADER_STORAGE_WRITE,
                ),
            ), // RaW
            (
                [
                    ReadWrite::Write(S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE),
                    ReadWrite::Write(S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE),
                ],
                (
                    S::COMPUTE_SHADER,
                    A::SHADER_STORAGE_WRITE,
                    S::COMPUTE_SHADER,
                    A::SHADER_STORAGE_WRITE,
                ),
            ), // WaW
            (
                [
                    ReadWrite::Write(S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE),
                    ReadWrite::Read(S::INDEX_INPUT, A::INDEX_READ),
                ],
                (
                    S::COMPUTE_SHADER,
                    A::SHADER_STORAGE_WRITE,
                    S::INDEX_INPUT,
                    A::INDEX_READ,
                ),
            ), // Dispatch writes into a storage buffer. Draw consumes that buffer as an index buffer.
        ];

        let mut buffer: vk::Buffer = unsafe { std::mem::transmute(123_u64) };
        for test_case in test_cases.into_iter() {
            let mut buffer = RenderRes::new(&mut buffer);
            let mut stage1 = make_stage(0);
            test_case.0[0].stage(&mut stage1, &mut buffer);

            let mut stage2 = make_stage(1);
            test_case.0[1].stage(&mut stage2, &mut buffer);

            let mut called = false;
            CommandBufferRecordContext::add_barrier(&stage2, |dep| {
                called = true;
                assert_global(
                    dep,
                    test_case.1 .0,
                    test_case.1 .1,
                    test_case.1 .2,
                    test_case.1 .3,
                );
            });
            if test_case.1 .0 != vk::PipelineStageFlags2::NONE
                && test_case.1 .2 != vk::PipelineStageFlags2::NONE
            {
                assert!(called);
            } else {
                assert!(!called);
            }
        }
    }

    #[test]
    fn c2c_global_carryover_tests() {
        let test_cases = [
            (
                [
                    ReadWrite::Write(S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE),
                    ReadWrite::ReadWrite(Access {
                        read_stages: S::COMPUTE_SHADER,
                        read_access: A::SHADER_STORAGE_READ,
                        write_stages: S::COMPUTE_SHADER,
                        write_access: A::SHADER_STORAGE_WRITE,
                    }),
                    ReadWrite::Read(S::COMPUTE_SHADER, A::SHADER_STORAGE_READ),
                ],
                [
                    (
                        S::COMPUTE_SHADER,
                        A::SHADER_STORAGE_WRITE,
                        S::COMPUTE_SHADER,
                        A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE,
                    ),
                    (
                        S::COMPUTE_SHADER,
                        A::SHADER_STORAGE_WRITE,
                        S::COMPUTE_SHADER,
                        A::SHADER_STORAGE_READ,
                    ),
                ],
            ),
            (
                [
                    ReadWrite::Read(S::COMPUTE_SHADER, A::SHADER_STORAGE_READ),
                    ReadWrite::ReadWrite(Access {
                        read_stages: S::COMPUTE_SHADER,
                        read_access: A::SHADER_STORAGE_READ,
                        write_stages: S::COMPUTE_SHADER,
                        write_access: A::SHADER_STORAGE_WRITE,
                    }),
                    ReadWrite::Write(S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE),
                ],
                [
                    (S::COMPUTE_SHADER, A::empty(), S::COMPUTE_SHADER, A::empty()),
                    (
                        S::COMPUTE_SHADER,
                        A::SHADER_STORAGE_WRITE,
                        S::COMPUTE_SHADER,
                        A::SHADER_STORAGE_WRITE,
                    ),
                ],
            ),
        ];
        let mut buffer: vk::Buffer = unsafe { std::mem::transmute(123_u64) };
        for test_case in test_cases.into_iter() {
            let mut buffer = RenderRes::new(&mut buffer);

            let mut stage1 = make_stage(0);
            test_case.0[0].stage(&mut stage1, &mut buffer);

            let mut stage2 = make_stage(1);
            test_case.0[1].stage(&mut stage2, &mut buffer);

            let mut called = false;
            CommandBufferRecordContext::add_barrier(&stage2, |dep| {
                called = true;
                assert_global(
                    dep,
                    test_case.1[0].0,
                    test_case.1[0].1,
                    test_case.1[0].2,
                    test_case.1[0].3,
                );
            });

            if test_case.1[0].0 != vk::PipelineStageFlags2::NONE
                && test_case.1[0].2 != vk::PipelineStageFlags2::NONE
            {
                assert!(called);
            }
            called = false;

            let mut stage3 = make_stage(2);
            test_case.0[2].stage(&mut stage3, &mut buffer);

            CommandBufferRecordContext::add_barrier(&stage3, |dep| {
                called = true;
                assert_global(
                    dep,
                    test_case.1[1].0,
                    test_case.1[1].1,
                    test_case.1[1].2,
                    test_case.1[1].3,
                );
            });
            if test_case.1[1].0 != vk::PipelineStageFlags2::NONE
                && test_case.1[1].2 != vk::PipelineStageFlags2::NONE
            {
                assert!(called);
            }
        }
    }

    #[test]
    fn c2c_image_tests() {
        let image: vk::Image = unsafe { std::mem::transmute(123_usize) };
        let mut stage_image = StageContextImage {
            image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: vk::REMAINING_MIP_LEVELS,
                base_array_layer: 0,
                layer_count: vk::REMAINING_ARRAY_LAYERS,
            },
            extent: vk::Extent3D::default(),
        };
        let image2: vk::Image = unsafe { std::mem::transmute(456_usize) };
        let mut stage_image2 = StageContextImage {
            image: image2,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: vk::REMAINING_MIP_LEVELS,
                base_array_layer: 0,
                layer_count: vk::REMAINING_ARRAY_LAYERS,
            },
            extent: vk::Extent3D::default(),
        };

        let mut buffer1: vk::Buffer = unsafe { std::mem::transmute(4562_usize) };
        let mut buffer2: vk::Buffer = unsafe { std::mem::transmute(578_usize) };

        {
            let mut stage_image_res = RenderImage::new(&mut stage_image, vk::ImageLayout::GENERAL);
            // First dispatch writes to a storage image, second dispatch reads from that storage image.
            let mut stage1 = make_stage(0);
            stage1.write_image(
                &mut stage_image_res,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::ImageLayout::GENERAL,
            );

            let mut stage2 = make_stage(1);
            stage2.read_image(
                &mut stage_image_res,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_READ,
                vk::ImageLayout::GENERAL,
            );

            let mut called = false;
            CommandBufferRecordContext::add_barrier(&stage2, |dep| {
                called = true;
                assert_global(
                    dep,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_STORAGE_WRITE,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_STORAGE_READ,
                );
            });
            assert!(called);
        }
        {
            let mut stage_image_res = RenderImage::new(&mut stage_image, vk::ImageLayout::GENERAL);
            // Dispatch writes into a storage image. Draw samples that image in a fragment shader.
            let mut stage1 = make_stage(0);
            stage1.write_image(
                &mut stage_image_res,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::ImageLayout::GENERAL,
            );

            let mut stage2 = make_stage(1);
            stage2.read_image(
                &mut stage_image_res,
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
                vk::AccessFlags2::SHADER_SAMPLED_READ,
                vk::ImageLayout::READ_ONLY_OPTIMAL,
            );

            let mut called = false;
            CommandBufferRecordContext::add_barrier(&stage2, |dep| {
                called = true;
                assert_eq!(dep.memory_barrier_count, 0);
                assert_eq!(dep.buffer_memory_barrier_count, 0);
                assert_eq!(dep.image_memory_barrier_count, 1);
                let image_memory_barrier = unsafe { &*dep.p_image_memory_barriers };
                assert_eq!(image_memory_barrier.old_layout, vk::ImageLayout::GENERAL);
                assert_eq!(
                    image_memory_barrier.new_layout,
                    vk::ImageLayout::READ_ONLY_OPTIMAL
                );
                assert_eq!(
                    image_memory_barrier.src_stage_mask,
                    vk::PipelineStageFlags2::COMPUTE_SHADER
                );
                assert_eq!(
                    image_memory_barrier.src_access_mask,
                    vk::AccessFlags2::SHADER_STORAGE_WRITE
                );
                assert_eq!(
                    image_memory_barrier.dst_stage_mask,
                    vk::PipelineStageFlags2::FRAGMENT_SHADER
                );
                assert_eq!(
                    image_memory_barrier.dst_access_mask,
                    vk::AccessFlags2::SHADER_SAMPLED_READ
                );
            });
            assert!(called);
        }

        {
            // Tests that image access info are retained across stages.
            let mut stage_image_res = RenderImage::new(&mut stage_image, vk::ImageLayout::GENERAL);
            let mut stage_image_res2 =
                RenderImage::new(&mut stage_image2, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
            let mut buffer_res1 = RenderRes::new(&mut buffer1);
            let mut buffer_res2 = RenderRes::new(&mut buffer2);
            // Stage 1 is a compute shader which writes into buffer1 and an image.
            // Stage 2 is a graphics pass which reads buffer1 as the vertex input and writes to another image.
            // Stage 3 is a compute shader, reads both images, and writes into buffer2.
            // Stage 4 is a compute shader that reads buffer2.
            let mut stage1 = make_stage(0);
            stage1.write(
                &mut buffer_res1,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
            );
            stage1.write_image(
                &mut stage_image_res,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::ImageLayout::GENERAL,
            );

            let mut stage2 = make_stage(1);
            stage2.read(
                &mut buffer_res1,
                vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT,
                vk::AccessFlags2::VERTEX_ATTRIBUTE_READ,
            );
            stage2.write_image(
                &mut stage_image_res2,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );

            let mut called = false;
            CommandBufferRecordContext::add_barrier(&stage2, |dep| {
                called = true;
                assert_global(
                    dep,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_STORAGE_WRITE,
                    vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT,
                    vk::AccessFlags2::VERTEX_ATTRIBUTE_READ,
                );
            });
            assert!(called);

            let mut stage3 = make_stage(2);
            stage3.read_image(
                &mut stage_image_res,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_READ,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
            stage3.read_image(
                &mut stage_image_res2,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_READ,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
            stage3.write(
                &mut buffer_res2,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
            );

            called = false;
            CommandBufferRecordContext::add_barrier(&stage3, |dep| {
                called = true;
                assert_eq!(dep.memory_barrier_count, 0);
                assert_eq!(dep.buffer_memory_barrier_count, 0);
                assert_eq!(dep.image_memory_barrier_count, 2);
                let image_memory_barriers = unsafe {
                    std::slice::from_raw_parts(
                        dep.p_image_memory_barriers,
                        dep.image_memory_barrier_count as usize,
                    )
                };
                {
                    let image_memory_barrier = image_memory_barriers
                        .iter()
                        .find(|a| a.image == image2)
                        .unwrap();
                    assert_eq!(
                        image_memory_barrier.old_layout,
                        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                    );
                    assert_eq!(
                        image_memory_barrier.new_layout,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                    );
                    assert_eq!(
                        image_memory_barrier.src_stage_mask,
                        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
                    );
                    assert_eq!(
                        image_memory_barrier.src_access_mask,
                        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                    );
                    assert_eq!(
                        image_memory_barrier.dst_stage_mask,
                        vk::PipelineStageFlags2::COMPUTE_SHADER
                    );
                    assert_eq!(
                        image_memory_barrier.dst_access_mask,
                        vk::AccessFlags2::SHADER_STORAGE_READ
                    );
                }
                {
                    let image_memory_barrier = image_memory_barriers
                        .iter()
                        .find(|a| a.image == image)
                        .unwrap();
                    assert_eq!(image_memory_barrier.old_layout, vk::ImageLayout::GENERAL);
                    assert_eq!(
                        image_memory_barrier.new_layout,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                    );
                    assert_eq!(
                        image_memory_barrier.src_stage_mask,
                        vk::PipelineStageFlags2::COMPUTE_SHADER
                    );
                    assert_eq!(
                        image_memory_barrier.src_access_mask,
                        vk::AccessFlags2::SHADER_STORAGE_WRITE
                    );
                    assert_eq!(
                        image_memory_barrier.dst_stage_mask,
                        vk::PipelineStageFlags2::COMPUTE_SHADER
                    );
                    assert_eq!(
                        image_memory_barrier.dst_access_mask,
                        vk::AccessFlags2::SHADER_STORAGE_READ
                    );
                }
            });
            assert!(called);

            let mut stage4 = make_stage(3);
            stage4.read(
                &mut buffer_res2,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            );
            called = false;
            CommandBufferRecordContext::add_barrier(&stage4, |dep| {
                called = true;
                assert_global(
                    dep,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_STORAGE_WRITE,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_STORAGE_READ,
                );
            });
            assert!(called);
        }
    }
}
