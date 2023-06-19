use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::{Debug, Write},
    marker::PhantomData,
    ops::Generator,
    pin::Pin,
    sync::Arc,
    task::Poll,
};

use super::compile::QueueCompileExt;
use ash::{prelude::VkResult, vk};

use pin_project::pin_project;

use crate::{
    commands::SharedCommandPool,
    future::{
        CommandBufferRecordContext, Disposable, GPUCommandFuture, StageContextBuffer,
        StageContextImage, StageContextSemaphoreTransition,
    },
    Device, HasDevice,
};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct QueueRef(pub u8);
impl QueueRef {
    pub fn null() -> Self {
        QueueRef(u8::MAX)
    }
    pub fn is_null(&self) -> bool {
        self.0 == u8::MAX
    }
}
impl Default for QueueRef {
    fn default() -> Self {
        Self::null()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct QueueMask(u64);
impl QueueMask {
    pub fn set_queue(&mut self, queue: QueueRef) {
        self.0 |= 1 << queue.0;
    }
    pub fn clear_queue(&mut self, queue: QueueRef) {
        self.0 &= !(1 << queue.0);
    }
    pub fn iter(&self) -> QueueMaskIterator {
        QueueMaskIterator(self.0)
    }
    pub fn empty() -> Self {
        Self(0)
    }
    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }
    pub fn merge_with(&mut self, other: Self) {
        self.0 |= other.0
    }
    pub fn merge(&self, other: &Self) -> Self {
        Self(self.0 | other.0)
    }
}
impl std::fmt::Debug for QueueMask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}
pub struct QueueMaskIterator(u64);
impl Iterator for QueueMaskIterator {
    type Item = QueueRef;
    fn next(&mut self) -> Option<Self::Item> {
        if self.0 == 0 {
            return None;
        }
        let t = self.0 & self.0.overflowing_neg().0;
        let r = self.0.trailing_zeros();
        self.0 ^= t;
        Some(QueueRef(r as u8))
    }
}

/// One for each submission.
pub struct SubmissionContext<'a> {
    // Indexed by queue family.
    pub shared_command_pools: &'a mut [Option<SharedCommandPool>],
    // Indexed by queue id.
    pub queues: Vec<QueueSubmissionContext>,
    // Indexed by queue id.
    pub submission: Vec<QueueSubmissionType>,
}

impl<'a> SubmissionContext<'a> {
    pub fn of_queue_mut(&mut self, queue: QueueRef) -> &mut QueueSubmissionContext {
        &mut self.queues[queue.0 as usize]
    }
}

pub enum QueueSubmissionContextSemaphoreWait {
    /// `dst_stages` of the current queue, must wait on the `src_stages` of `queue`.
    WaitForSignal {
        dst_stages: vk::PipelineStageFlags2,
        queue: QueueRef,
        src_stages: vk::PipelineStageFlags2,
    },
    /// `dst_stages` of the current queue must wait on the `acquire_semaphore`
    WaitForAcquire {
        dst_stages: vk::PipelineStageFlags2,
        acquire_semaphore: vk::Semaphore,
    },
}

pub(crate) enum QueueSubmissionContextExport {
    Image {
        image: StageContextImage,
        barrier: vk::MemoryBarrier2,
        dst_queue_family: u32,
        src_layout: vk::ImageLayout,
        dst_layout: vk::ImageLayout,
    },
    Buffer {
        buffer: StageContextBuffer,
        barrier: vk::MemoryBarrier2,
        dst_queue_family: u32,
    },
}

/// One per queue per submission
pub struct QueueSubmissionContext {
    pub queue_family_index: u32,
    pub stage_index: u32,

    /// Timeline index is defined as the timeline value signaled at the end of the stage.
    /// Therefore, it begins at 1.
    pub timeline_index: u32,

    /// Stages that the previous stage of the current queue needs to signal on.
    /// A list of (src_stages, force_binary)
    pub signals: BTreeSet<(vk::PipelineStageFlags2, bool)>,

    /// Stages that the current stage needs to wait on.
    /// A list of (dst_stage, src_queue, src_stage)
    pub waits: Vec<QueueSubmissionContextSemaphoreWait>,

    pub(crate) exports: Vec<QueueSubmissionContextExport>,
}

#[derive(Debug)]
pub enum QueueSubmissionType {
    Submit {
        command_buffers: Vec<vk::CommandBuffer>,
        recording_command_buffer: Option<vk::CommandBuffer>,
    },
    SparseBind {
        memory_binds: Vec<vk::SparseBufferMemoryBindInfo>,
        image_opaque_binds: Vec<vk::SparseImageOpaqueMemoryBindInfo>,
        image_binds: Vec<vk::SparseImageMemoryBindInfo>,
    },
    Acquire,
    Present(Vec<(vk::SwapchainKHR, u32)>),
    Unknown,
}
/// TODO: This is not Send because SparseBind is bad, it contains raw ptrs. Fix this.
unsafe impl Send for QueueSubmissionType {}
impl Default for QueueSubmissionType {
    fn default() -> Self {
        Self::Unknown
    }
}
impl QueueSubmissionType {
    pub fn end(&mut self, device: &crate::Device) {
        if let QueueSubmissionType::Submit {
            command_buffers,
            recording_command_buffer,
        } = self
        {
            if let Some(cmd_buf) = recording_command_buffer.take() {
                unsafe {
                    device.end_command_buffer(cmd_buf).unwrap();
                }
                command_buffers.push(cmd_buf);
            }
        }
    }
}
/// Queues returned by the device.
pub struct Queues {
    device: Arc<Device>,

    /// Queues, and their queue family index.
    queues: Vec<(vk::Queue, u32)>,

    /// Mapping from queue families to queue refs
    /// TODO: this field is unnecessary
    families: Vec<QueueMask>,
}

impl HasDevice for Queues {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl Queues {
    /// This function can only be called once per device.
    pub(crate) unsafe fn new(device: Arc<Device>) -> Self {
        let queues: Vec<(vk::Queue, u32)> = device
            .queue_info()
            .queues
            .iter()
            .map(|(family, index)| {
                let queue = unsafe { device.get_device_queue(*family, *index) };
                (queue, *family)
            })
            .collect();

        Self {
            device: device.clone(),
            queues,
            families: device.queue_info().families.clone(),
        }
    }
    pub fn submit<'a, F: QueueFuture>(
        &mut self,
        future: F,
        // These pools are passed in as argument so that they can be cleaned on a regular basis (per frame) externally.
        // The lifetime parameter prevents the caller from dropping the polls before awaiting the returned future.
        shared_command_pools: &mut [Option<SharedCommandPool>],
        semaphore_pool: &mut TimelineSemaphorePool,
        fence_pool: &mut FencePool,
        recycled_state: &mut F::RecycledState,
        apply_final_signal: bool,
    ) -> QueueSubmitFuture<F::RetainedState, F::Output>
    where
        F::Output: 'a,
        F::RetainedState: 'a,
    {
        let compiled = future.compile(
            shared_command_pools,
            semaphore_pool,
            recycled_state,
            apply_final_signal,
        );

        let mut submission_batch = SubmissionBatch::new(self.queues.len());
        for stage in compiled.submission_batch.into_iter() {
            submission_batch.add_stage(stage);
        }

        let fences_to_wait = self.submit_batch(submission_batch, fence_pool);

        let device = self.device.clone();
        QueueSubmitFuture::new(
            device,
            fences_to_wait,
            compiled.final_signals,
            compiled.fut_dispose,
            compiled.output,
        )
    }
    /// Safety: Users must manage lifetimes of command buffers and fences manually
    pub unsafe fn submit_compiled(
        &mut self,
        submissions: impl Iterator<Item = CachedStageSubmissions>,
        fence_pool: &mut impl FencePoolLike,
    ) -> Vec<vk::Fence> {
        let mut submission_batch = SubmissionBatch::new(self.queues.len());
        for stage in submissions {
            submission_batch.add_stage(stage);
        }

        let fences_to_wait = self.submit_batch(submission_batch, fence_pool);
        fences_to_wait
    }
    fn submit_batch(
        &mut self,
        batch: SubmissionBatch,
        fence_pool: &mut impl FencePoolLike,
    ) -> Vec<vk::Fence> {
        let mut fences: Vec<vk::Fence> = Vec::new();
        for (queue, queue_batch) in self
            .queues
            .iter()
            .map(|(queue, _)| *queue)
            .zip(batch.queues.iter())
        {
            unsafe {
                if !queue_batch.submits.is_empty() {
                    let fence = fence_pool.get();
                    self.device
                        .queue_submit2(queue, &queue_batch.submits, fence)
                        .unwrap();
                    fences.push(fence);
                }
                if !queue_batch.sparse_binds.is_empty() {
                    let fence = fence_pool.get();
                    self.device
                        .queue_bind_sparse(queue, &queue_batch.sparse_binds, fence)
                        .unwrap();
                    fences.push(fence);
                }
            }
        }

        // Presents after all others. This ensures that the binary semaphores used by the present operation
        // were submitted after all others.
        for (queue, queue_batch) in self
            .queues
            .iter()
            .map(|(queue, _)| *queue)
            .zip(batch.queues.iter())
        {
            unsafe {
                for info in queue_batch.presents.iter() {
                    /*
                    VK_EXT_swapchain_maintenance1
                    let fences: Vec<_> = (0..info.swapchain_count as usize)
                        .map(|_| {
                            let fence = fence_pool.get();
                            fences.push(fence);
                            fence
                        })
                        .collect();
                    let fence_info = vk::SwapchainPresentFenceInfoEXT {
                        swapchain_count: info.swapchain_count,
                        p_fences: fences.as_ptr(),
                        ..Default::default()
                    };
                    let present_info = vk::PresentInfoKHR {
                        p_next: &fence_info as *const _ as *const _,
                        ..*info
                    };
                    */
                    self.device
                        .swapchain_loader()
                        .queue_present(queue, &info)
                        .unwrap();
                }
            }
        }
        fences
    }
}

#[pin_project]
pub struct QueueSubmitFuture<Ret, Out> {
    #[pin]
    task: blocking::Task<VkResult<()>>,
    retained_state: Option<Ret>,
    output: Option<Out>,
    semaphores: Option<Vec<(vk::Semaphore, u64)>>,
}
impl<Ret, Out> QueueSubmitFuture<Ret, Out> {
    fn new(
        device: Arc<Device>,
        fences: Vec<vk::Fence>,
        semaphores: Option<Vec<(vk::Semaphore, u64)>>,
        retained_state: Ret,
        output: Out,
    ) -> Self {
        let task = blocking::unblock(move || unsafe { device.wait_for_fences(&fences, true, !0) });
        Self {
            task,
            semaphores,
            retained_state: Some(retained_state),
            output: Some(output),
        }
    }
}
impl<Ret: Disposable, Out> std::future::Future for QueueSubmitFuture<Ret, Out> {
    type Output = VkResult<Out>;

    fn poll(self: Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Self::Output> {
        let this = self.project();
        match this.task.poll(cx) {
            Poll::Pending => return Poll::Pending,
            Poll::Ready(err) => {
                this.retained_state.take().unwrap().dispose();
                return Poll::Ready(err.map(|_| this.output.take().unwrap()));
            }
        }
    }
}
impl<Ret: Disposable + Send, Out> QueueFuture for QueueSubmitFuture<Ret, Out> {
    type Output = VkResult<Out>;

    type RecycledState = ();

    type RetainedState = Ret;

    fn setup(
        self: Pin<&mut Self>,
        _ctx: &mut SubmissionContext,
        _recycled_state: &mut Self::RecycledState,
        _prev_queue: QueueMask,
    ) {
        let this = self.project();
        assert!(this.semaphores.is_some(), "To use a QueueSubmitFuture in another future, it must have been created with `apply_final_signal=true`");
    }

    fn record(
        self: Pin<&mut Self>,
        _ctx: &mut SubmissionContext,
        _recycled_state: &mut Self::RecycledState,
    ) -> QueueFuturePoll<Self::Output> {
        let this = self.project();
        if this.task.is_finished() {
            return QueueFuturePoll::Ready {
                next_queue: QueueMask::empty(),
                output: Ok(this.output.take().unwrap()), // TODO: Check err
            };
        }

        QueueFuturePoll::Semaphore(std::mem::take(this.semaphores.as_mut().unwrap()))
    }

    fn dispose(mut self) -> Self::RetainedState {
        // TODO: There might be a problem with retire() being called twice.
        self.retained_state.take().unwrap()
    }
}
impl<Ret: Disposable + Send, Out> QueueSubmitFuture<Ret, Out> {
    pub fn shared(
        self,
    ) -> (
        SharedQueueSubmitFutureMain<Ret, Out>,
        SharedQueueSubmitFuture<Out>,
    ) {
        let main = SharedQueueSubmitFutureMain {
            task: Arc::new(self.task),
            retained_state: self.retained_state,
            output: Arc::new(self.output.unwrap()),
            semaphores: self.semaphores.expect("To use a QueueSubmitFuture as shared, it must have been created with `apply_final_signal=true`"),
        };
        let shared = SharedQueueSubmitFuture {
            task: main.task.clone(),
            output: main.output.clone(),
        };
        (main, shared)
    }
}

#[pin_project]
pub struct SharedQueueSubmitFutureMain<Ret, Out> {
    #[pin]
    task: Arc<blocking::Task<VkResult<()>>>,
    retained_state: Option<Ret>,
    output: Arc<Out>,
    semaphores: Vec<(vk::Semaphore, u64)>,
}
impl<Ret: Disposable + Send, Out> QueueFuture for SharedQueueSubmitFutureMain<Ret, Out> {
    type Output = Arc<Out>;

    type RecycledState = ();

    type RetainedState = Ret;

    fn setup(
        self: Pin<&mut Self>,
        _ctx: &mut SubmissionContext,
        _recycled_state: &mut Self::RecycledState,
        _prev_queue: QueueMask,
    ) {
    }

    fn record(
        self: Pin<&mut Self>,
        _ctx: &mut SubmissionContext,
        _recycled_state: &mut Self::RecycledState,
    ) -> QueueFuturePoll<Self::Output> {
        let this = self.project();
        if this.task.is_finished() {
            // TODO: destroy semaphores
            return QueueFuturePoll::Ready {
                next_queue: QueueMask::empty(),
                output: this.output.clone(),
            };
        }

        QueueFuturePoll::Semaphore(std::mem::take(this.semaphores))
    }

    fn dispose(mut self) -> Self::RetainedState {
        self.retained_state.take().unwrap()
    }
}

#[derive(Clone)]
#[pin_project]
pub struct SharedQueueSubmitFuture<Out> {
    #[pin]
    task: Arc<blocking::Task<VkResult<()>>>,
    output: Arc<Out>,
}
impl<Out> QueueFuture for SharedQueueSubmitFuture<Out> {
    type Output = Arc<Out>;

    type RecycledState = ();

    type RetainedState = ();

    fn setup(
        self: Pin<&mut Self>,
        _ctx: &mut SubmissionContext,
        _recycled_state: &mut Self::RecycledState,
        _prev_queue: QueueMask,
    ) {
    }

    fn record(
        self: Pin<&mut Self>,
        _ctx: &mut SubmissionContext,
        _recycled_state: &mut Self::RecycledState,
    ) -> QueueFuturePoll<Self::Output> {
        let this = self.project();
        assert!(this.task.is_finished());
        return QueueFuturePoll::Ready {
            next_queue: QueueMask::empty(),
            output: this.output.clone(),
        };
    }

    fn dispose(self) -> Self::RetainedState {
        ()
    }
}

/// Resource pool to be recreated for each submission.
pub struct TimelineSemaphorePool {
    device: Arc<Device>,
    timeline_semaphores: Vec<vk::Semaphore>,
    semaphore_ops: Vec<(vk::Semaphore, u64)>,

    binary_semaphores: Vec<vk::Semaphore>,
    binary_indice: usize,
}
impl HasDevice for TimelineSemaphorePool {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}
impl TimelineSemaphorePool {
    pub fn new(device: Arc<Device>) -> Self {
        Self {
            device,
            timeline_semaphores: Vec::new(),
            semaphore_ops: Vec::new(),
            binary_semaphores: Vec::new(),
            binary_indice: 0,
        }
    }
    pub fn signal(&mut self) -> (vk::Semaphore, u64) {
        let (semaphore, timeline) = self.semaphore_ops.pop().unwrap_or_else(|| {
            let mut type_info = vk::SemaphoreTypeCreateInfo::builder()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0)
                .build();
            let create_info = vk::SemaphoreCreateInfo::builder()
                .push_next(&mut type_info)
                .build();
            let semaphore = unsafe { self.device.create_semaphore(&create_info, None).unwrap() };
            self.timeline_semaphores.push(semaphore);
            (semaphore, 0)
        });
        (semaphore, timeline + 1)
    }
    pub fn waited(&mut self, semaphore: vk::Semaphore, timeline: u64) {
        self.semaphore_ops.push((semaphore, timeline));
    }
    pub fn get_binary_semaphore(&mut self) -> vk::Semaphore {
        if self.binary_indice >= self.binary_semaphores.len() {
            let semaphore = unsafe {
                self.device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                    .unwrap()
            };
            self.binary_semaphores.push(semaphore);
        }
        let semaphore = self.binary_semaphores[self.binary_indice];
        self.binary_indice += 1;
        semaphore
    }
    pub fn reset(&mut self) {
        assert_eq!(self.timeline_semaphores.len(), self.semaphore_ops.len());
        for semaphore in self.timeline_semaphores.drain(..) {
            unsafe {
                self.device.destroy_semaphore(semaphore, None);
            }
        }
        self.binary_indice = 0;
        self.semaphore_ops.clear();
    }
}
impl Drop for TimelineSemaphorePool {
    fn drop(&mut self) {
        for semaphore in self.timeline_semaphores.iter() {
            unsafe {
                self.device.destroy_semaphore(*semaphore, None);
            }
        }

        for semaphore in self.binary_semaphores.iter() {
            unsafe {
                self.device.destroy_semaphore(*semaphore, None);
            }
        }
    }
}

/// Resource pool to be reset each frame.
pub struct FencePool {
    device: Arc<Device>,
    all_fences: Vec<vk::Fence>,
    indice: usize,
}
impl FencePool {
    pub fn new(device: Arc<Device>) -> Self {
        Self {
            device,
            all_fences: Vec::new(),
            indice: 0,
        }
    }
    pub fn get(&mut self) -> vk::Fence {
        if self.indice >= self.all_fences.len() {
            let fence = unsafe {
                self.device
                    .create_fence(&vk::FenceCreateInfo::default(), None)
                    .unwrap()
            };
            self.all_fences.push(fence);
        }
        let fence = self.all_fences[self.indice];
        self.indice += 1;
        fence
    }
    pub fn reset(&mut self) {
        if self.indice > 0 {
            unsafe {
                self.device
                    .reset_fences(&self.all_fences[0..self.indice])
                    .unwrap()
            }
        }
        self.indice = 0;
    }
}
impl Drop for FencePool {
    fn drop(&mut self) {
        for fence in self.all_fences.iter() {
            unsafe {
                self.device.destroy_fence(*fence, None);
            }
        }
    }
}

pub trait FencePoolLike {
    fn get(&mut self) -> vk::Fence;
}
impl FencePoolLike for FencePool {
    fn get(&mut self) -> vk::Fence {
        self.get()
    }
}

/// Represents one queue of one stage of submissions
#[derive(Default, Debug)]
pub(super) struct CachedQueueStageSubmissions {
    // Indexed
    pub ty: QueueSubmissionType,
    pub(super) waits: Vec<(vk::Semaphore, u64, vk::PipelineStageFlags2)>,
    pub(super) signals: BTreeMap<vk::PipelineStageFlags2, (vk::Semaphore, u64)>,
}
impl CachedQueueStageSubmissions {
    pub fn apply_exports(&mut self, ctx: &QueueSubmissionContext, device: &Device) {
        let mut image_barriers: Vec<vk::ImageMemoryBarrier2> = Vec::new();
        let mut buffer_barriers: Vec<vk::BufferMemoryBarrier2> = Vec::new();

        for export in ctx.exports.iter() {
            match export {
                QueueSubmissionContextExport::Image {
                    image,
                    barrier,
                    dst_queue_family,
                    src_layout,
                    dst_layout,
                } => {
                    image_barriers.push(vk::ImageMemoryBarrier2 {
                        src_stage_mask: barrier.src_stage_mask,
                        src_access_mask: barrier.src_access_mask,
                        dst_access_mask: barrier.dst_access_mask,
                        dst_stage_mask: barrier.dst_stage_mask,
                        old_layout: *src_layout,
                        new_layout: *dst_layout,
                        src_queue_family_index: ctx.queue_family_index,
                        dst_queue_family_index: *dst_queue_family,
                        image: image.image,
                        subresource_range: image.subresource_range.clone(),
                        ..Default::default()
                    });
                }
                QueueSubmissionContextExport::Buffer {
                    buffer,
                    barrier,
                    dst_queue_family,
                } => {
                    buffer_barriers.push(vk::BufferMemoryBarrier2 {
                        src_stage_mask: barrier.src_stage_mask,
                        src_access_mask: barrier.src_access_mask,
                        dst_access_mask: barrier.dst_access_mask,
                        dst_stage_mask: barrier.dst_stage_mask,
                        src_queue_family_index: ctx.queue_family_index,
                        dst_queue_family_index: *dst_queue_family,
                        buffer: buffer.buffer,
                        size: buffer.size,
                        offset: buffer.offset,
                        ..Default::default()
                    });
                }
                _ => panic!(),
            }
        }
        if image_barriers.is_empty() && buffer_barriers.is_empty() {
            return;
        }
        let dependency_info = vk::DependencyInfo::builder()
            .dependency_flags(vk::DependencyFlags::BY_REGION)
            .image_memory_barriers(&image_barriers)
            .buffer_memory_barriers(&buffer_barriers)
            .build();
        match &self.ty {
            QueueSubmissionType::Submit {
                recording_command_buffer: Some(command_buffer),
                ..
            } => unsafe {
                device.cmd_pipeline_barrier2(*command_buffer, &dependency_info);
            },
            _ => panic!(),
        }
    }
}
/// Represents one stage of submissions
#[derive(Debug)]
pub struct CachedStageSubmissions {
    // Indexed by QueueId
    pub(super) queues: Vec<CachedQueueStageSubmissions>,
}
impl CachedStageSubmissions {
    pub fn new(num_queues: usize) -> Self {
        Self {
            queues: (0..num_queues)
                .map(|_| CachedQueueStageSubmissions::default())
                .collect(),
        }
    }
    /// Called on the previous stage.
    pub fn apply_signals(
        &mut self,
        submission_context: &SubmissionContext,
        semaphore_pool: &mut TimelineSemaphorePool,
    ) {
        for (ctx, cache) in submission_context.queues.iter().zip(self.queues.iter_mut()) {
            cache
                .signals
                .extend(ctx.signals.iter().map(|(stage, force_binary)| {
                    if *force_binary {
                        let semaphore = semaphore_pool.get_binary_semaphore();
                        (*stage, (semaphore, u64::MAX))
                    } else {
                        let (semaphore, value) = semaphore_pool.signal();
                        (*stage, (semaphore, value))
                    }
                }));
        }
    }

    /// On the final stage of a QueueFuture, we can call this method to have each queue to signal one
    /// additional timeline semaphore.
    pub fn apply_final_signals(
        &mut self,
        submission_context: &SubmissionContext,
        semaphore_pool: &mut TimelineSemaphorePool,
    ) -> Vec<(vk::Semaphore, u64)> {
        submission_context
            .queues
            .iter()
            .zip(self.queues.iter_mut())
            .filter_map(|(ctx, cache)| {
                if ctx.stage_index == 0 {
                    // Didn't actually do anything in this queue.
                    return None;
                }
                // TODO: Figure out the right pipeline stages
                let signal = semaphore_pool.signal();
                cache
                    .signals
                    .insert(vk::PipelineStageFlags2::ALL_COMMANDS, signal);
                return Some(signal);
            })
            .collect()
    }
    /// The timeline semaphores signaled in `apply_final_signals` will be awaited here.
    pub fn wait_additional_signals(
        &mut self,
        submission_context: &SubmissionContext,
        semaphores: Vec<(vk::Semaphore, u64)>,
    ) {
        for (ctx, cache) in submission_context.queues.iter().zip(self.queues.iter_mut()) {
            if ctx.stage_index == 0 {
                // Didn't actually do anything in this queue.
                continue;
            }
            // TODO: Figure out the right pipeline stages
            cache
                .waits
                .extend(semaphores.iter().map(|(semaphore, timeline)| {
                    (*semaphore, *timeline, vk::PipelineStageFlags2::ALL_COMMANDS)
                }));
        }
        // TODO: destroy these semaphores
    }
    /// Called on the current stage.
    pub fn apply_submissions(
        &mut self,
        submission_context: &SubmissionContext,
        prev_stage: &Self,
        semaphore_pool: &mut TimelineSemaphorePool,
    ) {
        for (_i, (ctx, cache)) in submission_context
            .queues
            .iter()
            .zip(self.queues.iter_mut())
            .enumerate()
        {
            cache.waits.extend(ctx.waits.iter().map(|wait| {
                match wait {
                    QueueSubmissionContextSemaphoreWait::WaitForSignal {
                        dst_stages,
                        queue,
                        src_stages,
                    } => {
                        let (semaphore, value) = prev_stage.queues[queue.0 as usize]
                            .signals
                            .get(src_stages)
                            .unwrap();
                        if *value != u64::MAX {
                            // NOTE: Here's a potential issue: is it possible for the same semaphore to be `waited` on multiple times?
                            semaphore_pool.waited(*semaphore, *value);

                            (*semaphore, *value, *dst_stages)
                        } else {
                            // when value == u64::MAX, it's a binary semaphore.
                            // the `value` here gets fed into the `value` field of vk::SemaphoreSubmitInfo
                            // When semaphore is a binary semaphore, this value was ignored.
                            // We can also just return *value here, but to be safe let's set it to 0
                            (*semaphore, 0, *dst_stages)
                        }
                    }
                    QueueSubmissionContextSemaphoreWait::WaitForAcquire {
                        dst_stages,
                        acquire_semaphore,
                    } => (*acquire_semaphore, 0, *dst_stages),
                }
            }));
        }
    }
}

#[derive(Default)]
pub struct QueueSubmissionBatch {
    submits: Vec<vk::SubmitInfo2>,
    sparse_binds: Vec<vk::BindSparseInfo>,
    presents: Vec<vk::PresentInfoKHR>,
}
pub struct SubmissionBatch {
    queues: Vec<QueueSubmissionBatch>,
}
impl Debug for SubmissionBatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        struct FormatStages(vk::PipelineStageFlags2);
        impl Debug for FormatStages {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                if self.0.is_empty() {
                    f.write_str("<empty>")
                } else {
                    Debug::fmt(&self.0, f)
                }
            }
        }
        f.write_str("SubmissionBatch {")?;

        for (i, queue) in self.queues.iter().enumerate() {
            if !queue.submits.is_empty() {
                f.write_char('\n')?;
                f.write_fmt(format_args!("  Queue {i} Submits:\n"))?;
                for submit in queue.submits.iter() {
                    unsafe {
                        for i in 0..submit.wait_semaphore_info_count {
                            let semaphore = &*submit.p_wait_semaphore_infos.add(i as usize);
                            f.write_fmt(format_args!(
                                "    Stages {:?} wait semaphore {:?} on {:?}\n",
                                FormatStages(semaphore.stage_mask),
                                semaphore.semaphore,
                                semaphore.value
                            ))?;
                        }
                    }
                    f.write_fmt(format_args!(
                        "    Submit {} command buffers\n",
                        submit.command_buffer_info_count
                    ))?;
                    unsafe {
                        for i in 0..submit.signal_semaphore_info_count {
                            let semaphore = &*submit.p_signal_semaphore_infos.add(i as usize);
                            f.write_fmt(format_args!(
                                "    Stages {:?} signal semaphore {:?} on {:?}\n",
                                FormatStages(semaphore.stage_mask),
                                semaphore.semaphore,
                                semaphore.value
                            ))?;
                        }
                    }
                }
            }
            if !queue.presents.is_empty() {
                f.write_char('\n')?;
                f.write_fmt(format_args!("  Queue {i} Presents:\n"))?;
                for present in queue.presents.iter() {
                    unsafe {
                        for i in 0..present.wait_semaphore_count {
                            let semaphore = &*present.p_wait_semaphores.add(i as usize);
                            f.write_fmt(format_args!("    Wait semaphore {:?}\n", semaphore,))?;
                        }
                    }
                    f.write_fmt(format_args!(
                        "    Present {} swap chains\n",
                        present.swapchain_count
                    ))?;
                }
            }
        }
        f.write_str("}")?;
        Ok(())
    }
}
impl SubmissionBatch {
    pub fn new(num_queues: usize) -> Self {
        Self {
            queues: (0..num_queues).map(|_| Default::default()).collect(),
        }
    }
    fn add_stage(&mut self, stage: CachedStageSubmissions) {
        for (ctx, self_ctx) in stage.queues.into_iter().zip(self.queues.iter_mut()) {
            match ctx.ty {
                QueueSubmissionType::Submit {
                    command_buffers,
                    recording_command_buffer,
                } => {
                    assert!(recording_command_buffer.is_none());
                    let waits = ctx
                        .waits
                        .into_iter()
                        .map(|(semaphore, value, stage_mask)| vk::SemaphoreSubmitInfo {
                            semaphore,
                            value,
                            stage_mask,
                            ..Default::default()
                        })
                        .collect::<Vec<_>>()
                        .into_boxed_slice();
                    let signals = ctx
                        .signals
                        .into_iter()
                        .map(|(stage_mask, (semaphore, value))| vk::SemaphoreSubmitInfo {
                            semaphore,
                            value,
                            stage_mask,
                            ..Default::default()
                        })
                        .collect::<Vec<_>>()
                        .into_boxed_slice();
                    let command_buffers = command_buffers
                        .into_iter()
                        .map(|buf| vk::CommandBufferSubmitInfo {
                            command_buffer: buf,
                            ..Default::default()
                        })
                        .collect::<Vec<_>>()
                        .into_boxed_slice();
                    self_ctx.submits.push(
                        vk::SubmitInfo2::builder()
                            .command_buffer_infos(&command_buffers)
                            .wait_semaphore_infos(&waits)
                            .signal_semaphore_infos(&signals)
                            .build(),
                    );
                    std::mem::forget(waits);
                    std::mem::forget(signals);
                    std::mem::forget(command_buffers);
                }
                QueueSubmissionType::SparseBind {
                    memory_binds,
                    image_opaque_binds,
                    image_binds,
                } => {
                    for (_, _, stages) in ctx.waits.iter() {
                        assert!(stages.is_empty());
                    }
                    for (stages, _) in ctx.signals.iter() {
                        assert!(stages.is_empty());
                    }
                    let waits = ctx
                        .waits
                        .iter()
                        .map(|(semaphore, _, _)| *semaphore)
                        .collect::<Vec<_>>()
                        .into_boxed_slice();
                    let signals = ctx
                        .signals
                        .iter()
                        .map(|(_, (semaphore, _))| *semaphore)
                        .collect::<Vec<_>>()
                        .into_boxed_slice();
                    let wait_values = ctx
                        .waits
                        .iter()
                        .map(|(_, value, _)| *value)
                        .collect::<Vec<_>>()
                        .into_boxed_slice();
                    let signal_values = ctx
                        .signals
                        .iter()
                        .map(|(_, (_, value))| *value)
                        .collect::<Vec<_>>()
                        .into_boxed_slice();

                    let memory_binds = memory_binds.into_boxed_slice();
                    let image_opaque_binds = image_opaque_binds.into_boxed_slice();
                    let image_binds = image_binds.into_boxed_slice();
                    let mut timeline_info = Box::new(
                        vk::TimelineSemaphoreSubmitInfo::builder()
                            .wait_semaphore_values(&wait_values)
                            .signal_semaphore_values(&signal_values)
                            .build(),
                    );
                    self_ctx.sparse_binds.push(
                        vk::BindSparseInfo::builder()
                            .buffer_binds(&memory_binds)
                            .image_binds(&image_binds)
                            .image_opaque_binds(&image_opaque_binds)
                            .wait_semaphores(&waits)
                            .signal_semaphores(&signals)
                            .push_next(timeline_info.as_mut())
                            .build(),
                    );

                    std::mem::forget(waits);
                    std::mem::forget(signals);
                    std::mem::forget(wait_values);
                    std::mem::forget(signal_values);
                    std::mem::forget(memory_binds);
                    std::mem::forget(image_opaque_binds);
                    std::mem::forget(image_binds);
                    std::mem::forget(timeline_info);
                }
                QueueSubmissionType::Present(presents) => {
                    let waits = ctx
                        .waits
                        .iter()
                        .map(|(semaphore, value, stage)| {
                            assert_eq!(*value, 0);
                            assert!(stage.is_empty());
                            *semaphore
                        })
                        .collect::<Vec<_>>()
                        .into_boxed_slice();
                    assert!(ctx.signals.is_empty());
                    let swapchains: Box<[vk::SwapchainKHR]> = presents
                        .iter()
                        .map(|(swapchain, _)| *swapchain)
                        .collect::<Vec<_>>()
                        .into_boxed_slice();
                    let indices: Box<[u32]> = presents
                        .iter()
                        .map(|(_, indice)| *indice)
                        .collect::<Vec<_>>()
                        .into_boxed_slice();
                    self_ctx.presents.push(vk::PresentInfoKHR {
                        wait_semaphore_count: waits.len() as u32,
                        p_wait_semaphores: waits.as_ptr(),
                        swapchain_count: swapchains.len() as u32,
                        p_swapchains: swapchains.as_ptr(),
                        p_image_indices: indices.as_ptr(),
                        p_results: std::ptr::null_mut(),
                        ..Default::default()
                    });
                    std::mem::forget(waits);
                    std::mem::forget(swapchains);
                    std::mem::forget(indices);
                }
                QueueSubmissionType::Acquire => todo!(),
                QueueSubmissionType::Unknown => (),
            }
        }
    }
}
impl Drop for SubmissionBatch {
    fn drop(&mut self) {
        unsafe {
            for queue in self.queues.iter() {
                for submit in &queue.submits {
                    let waits = Box::from_raw(std::slice::from_raw_parts_mut(
                        submit.p_wait_semaphore_infos as *mut vk::SemaphoreSubmitInfo,
                        submit.wait_semaphore_info_count as usize,
                    ));
                    let signals = Box::from_raw(std::slice::from_raw_parts_mut(
                        submit.p_signal_semaphore_infos as *mut vk::SemaphoreSubmitInfo,
                        submit.signal_semaphore_info_count as usize,
                    ));
                    let commands = Box::from_raw(std::slice::from_raw_parts_mut(
                        submit.p_command_buffer_infos as *mut vk::SemaphoreSubmitInfo,
                        submit.command_buffer_info_count as usize,
                    ));
                    drop(waits);
                    drop(signals);
                    drop(commands);
                }
                for sparse_bind in &queue.sparse_binds {
                    drop(Box::from_raw(std::slice::from_raw_parts_mut(
                        sparse_bind.p_wait_semaphores as *mut vk::Semaphore,
                        sparse_bind.wait_semaphore_count as usize,
                    )));
                    drop(Box::from_raw(std::slice::from_raw_parts_mut(
                        sparse_bind.signal_semaphore_count as *mut vk::SemaphoreSubmitInfo,
                        sparse_bind.signal_semaphore_count as usize,
                    )));
                    drop(Box::from_raw(std::slice::from_raw_parts_mut(
                        sparse_bind.p_buffer_binds as *mut vk::SemaphoreSubmitInfo,
                        sparse_bind.buffer_bind_count as usize,
                    )));
                    drop(Box::from_raw(std::slice::from_raw_parts_mut(
                        sparse_bind.p_image_binds as *mut vk::SemaphoreSubmitInfo,
                        sparse_bind.image_bind_count as usize,
                    )));
                    drop(Box::from_raw(std::slice::from_raw_parts_mut(
                        sparse_bind.p_image_opaque_binds as *mut vk::SemaphoreSubmitInfo,
                        sparse_bind.image_opaque_bind_count as usize,
                    )));
                    let timeline =
                        Box::from_raw(sparse_bind.p_next as *mut vk::TimelineSemaphoreSubmitInfo);
                    drop(Box::from_raw(std::slice::from_raw_parts_mut(
                        timeline.p_signal_semaphore_values as *mut vk::SemaphoreSubmitInfo,
                        timeline.signal_semaphore_value_count as usize,
                    )));
                    drop(Box::from_raw(std::slice::from_raw_parts_mut(
                        timeline.p_wait_semaphore_values as *mut vk::SemaphoreSubmitInfo,
                        timeline.wait_semaphore_value_count as usize,
                    )));
                    drop(timeline);
                }
                for present in &queue.presents {
                    drop(Box::from_raw(std::slice::from_raw_parts_mut(
                        present.p_swapchains as *mut vk::SwapchainKHR,
                        present.swapchain_count as usize,
                    )));
                    drop(Box::from_raw(std::slice::from_raw_parts_mut(
                        present.p_image_indices as *mut u32,
                        present.swapchain_count as usize,
                    )));
                    drop(Box::from_raw(std::slice::from_raw_parts_mut(
                        present.p_wait_semaphores as *mut vk::Semaphore,
                        present.wait_semaphore_count as usize,
                    )));
                }
            }
        }
    }
}

pub enum QueueFuturePoll<OUT> {
    Barrier,
    /// Contains a list of additional semaphores to wait.
    Semaphore(Vec<(vk::Semaphore, u64)>),
    Ready {
        next_queue: QueueMask,
        output: OUT,
    },
}

/// We don't know what are the semaphores to signal, until later stages tell us.
/// When future2 depends on future1, the following execution order occures:
/// - future1.init()
/// - let ctx = future1.context();
/// - future1.record().......
///
///
/// - future2.init()
/// - let ctx = future2.context();
/// - future1.end(ctx)
/// - future2.record().......
pub trait QueueFuture {
    type Output;
    type RecycledState: Default + Send + Sync;
    type RetainedState: Disposable + Send;
    fn setup(
        self: Pin<&mut Self>,
        ctx: &mut SubmissionContext,
        recycled_state: &mut Self::RecycledState,
        prev_queue: QueueMask,
    );
    /// Record all command buffers for the specified queue_index, up to the specified timeline index.
    /// The executor calls record with increasing `timeline` value, and wrap them in vk::SubmitInfo2.
    /// queue should be the queue of the parent node, or None if multiple parents with different queues.
    /// queue should be None on subsequent calls.
    fn record(
        self: Pin<&mut Self>,
        ctx: &mut SubmissionContext,
        recycled_state: &mut Self::RecycledState,
    ) -> QueueFuturePoll<Self::Output>;

    /// Runs when the future is ready to be disposed.s
    fn dispose(self) -> Self::RetainedState;
}

/// On yield: true for a hard sync point (semaphore)
pub trait QueueFutureBlockGenerator<Return, RecycledState, RetainedState> = Generator<
    (QueueMask, *mut (), *mut RecycledState),
    Return = (QueueMask, RetainedState, Return),
    Yield = Option<Vec<(vk::Semaphore, u64)>>,
>;

#[pin_project]
pub struct QueueFutureBlock<Ret, Inner, Recycle, Retain>
where
    Inner: QueueFutureBlockGenerator<Ret, Recycle, Retain>,
{
    #[pin]
    inner: Inner,
    retained_state: Option<Retain>,

    /// This is more of a hack to save the dependent queue mask when `init` was called
    /// so we can initialize `__current_queue_mask` with this value.
    initial_queue_mask: QueueMask,
    _marker: PhantomData<fn(&mut Recycle) -> Ret>,
}
impl<Ret, Inner, Fut, Recycle> QueueFutureBlock<Ret, Inner, Fut, Recycle>
where
    Inner: QueueFutureBlockGenerator<Ret, Fut, Recycle>,
{
    pub fn new(inner: Inner) -> Self {
        Self {
            inner,
            initial_queue_mask: QueueMask::empty(),
            retained_state: None,
            _marker: PhantomData,
        }
    }
}
impl<Ret, Inner, Retain: Disposable + Send, Recycle: Default + Send + Sync> QueueFuture
    for QueueFutureBlock<Ret, Inner, Recycle, Retain>
where
    Inner: QueueFutureBlockGenerator<Ret, Recycle, Retain>,
{
    type Output = Ret;
    type RecycledState = Recycle;
    type RetainedState = Retain;
    fn setup(
        self: Pin<&mut Self>,
        _ctx: &mut SubmissionContext,
        _recycled_state: &mut Self::RecycledState,
        prev_queue: QueueMask,
    ) {
        *self.project().initial_queue_mask = prev_queue;
    }
    fn record(
        self: Pin<&mut Self>,
        ctx: &mut SubmissionContext,
        recycled_state: &mut Self::RecycledState,
    ) -> QueueFuturePoll<Ret> {
        let this = self.project();
        assert!(
            this.retained_state.is_none(),
            "Calling record after returning Complete"
        );
        match this.inner.resume((
            *this.initial_queue_mask,
            ctx as *mut _ as *mut (),
            recycled_state,
        )) {
            std::ops::GeneratorState::Yielded(is_semaphore) => {
                if let Some(additional_semaphores) = is_semaphore {
                    QueueFuturePoll::Semaphore(additional_semaphores)
                } else {
                    QueueFuturePoll::Barrier
                }
            }
            std::ops::GeneratorState::Complete((next_queue, dispose, output)) => {
                *this.retained_state = Some(dispose);
                QueueFuturePoll::Ready { next_queue, output }
            }
        }
    }
    fn dispose(mut self) -> Retain {
        self.retained_state.take().unwrap()
    }
}

/*
#[pin_project]
pub struct QueueFutureJoin<I1: QueueFuture, I2: QueueFuture> {
    #[pin]
    inner1: I1,
    inner1_result: QueueFuturePoll<I1::Output>,
    #[pin]
    inner2: I2,
    inner2_result: QueueFuturePoll<I2::Output>,
    results_taken: bool,
}

impl<I1: QueueFuture, I2: QueueFuture> QueueFutureJoin<I1, I2> {
    pub fn new(inner1: I1, inner2: I2) -> Self {
        Self {
            inner1,
            inner1_result: QueueFuturePoll::Barrier,
            inner2,
            inner2_result: QueueFuturePoll::Barrier,
            results_taken: false,
        }
    }
}

impl<I1: QueueFuture, I2: QueueFuture> QueueFuture for QueueFutureJoin<I1, I2> {
    type Output = (I1::Output, I2::Output);
    type RecycledState = (I1::RecycledState, I2::RecycledState);
    fn init(self: Pin<&mut Self>, ctx: &mut SubmissionContext, recycled_state: &mut Self::RecycledState, prev_queue: QueueMask) {
        let this = self.project();
        this.inner1.init(ctx, &mut recycled_state.0, prev_queue);
        this.inner2.init(ctx,  &mut recycled_state.1, prev_queue);
    }

    fn record(self: Pin<&mut Self>, ctx: &mut SubmissionContext, recycled_state: &mut Self::RecycledState) -> QueueFuturePoll<Self::Output> {
        let this = self.project();
        assert!(
            !*this.results_taken,
            "Attempted to record a QueueFutureJoin after it's finished"
        );
        match (&this.inner1_result, &this.inner2_result) {
            (QueueFuturePoll::Barrier, QueueFuturePoll::Barrier)
            | (QueueFuturePoll::Semaphore, QueueFuturePoll::Semaphore) => {
                *this.inner1_result = this.inner1.record(ctx, &mut recycled_state.0);
                *this.inner2_result = this.inner2.record(ctx, &mut recycled_state.1);
            }
            (QueueFuturePoll::Barrier, QueueFuturePoll::Semaphore)
            | (QueueFuturePoll::Barrier, QueueFuturePoll::Ready { .. })
            | (QueueFuturePoll::Semaphore, QueueFuturePoll::Ready { .. }) => {
                *this.inner1_result = this.inner1.record(ctx, &mut recycled_state.0);
            }
            (QueueFuturePoll::Semaphore, QueueFuturePoll::Barrier)
            | (QueueFuturePoll::Ready { .. }, QueueFuturePoll::Barrier)
            | (QueueFuturePoll::Ready { .. }, QueueFuturePoll::Semaphore) => {
                *this.inner2_result = this.inner2.record(ctx, &mut recycled_state.1);
            }
            (QueueFuturePoll::Ready { .. }, QueueFuturePoll::Ready { .. }) => {
                unreachable!();
            }
        }
        match (&this.inner1_result, &this.inner2_result) {
            (QueueFuturePoll::Barrier, QueueFuturePoll::Barrier)
            | (QueueFuturePoll::Barrier, QueueFuturePoll::Semaphore)
            | (QueueFuturePoll::Barrier, QueueFuturePoll::Ready { .. })
            | (QueueFuturePoll::Semaphore, QueueFuturePoll::Barrier)
            | (QueueFuturePoll::Ready { .. }, QueueFuturePoll::Barrier) => QueueFuturePoll::Barrier,
            (QueueFuturePoll::Semaphore, QueueFuturePoll::Semaphore)
            | (QueueFuturePoll::Ready { .. }, QueueFuturePoll::Semaphore)
            | (QueueFuturePoll::Semaphore, QueueFuturePoll::Ready { .. }) => {
                QueueFuturePoll::Semaphore
            }
            (QueueFuturePoll::Ready { .. }, QueueFuturePoll::Ready { .. }) => {
                let (r1_queue_mask, r1_ret) =
                    match std::mem::replace(this.inner1_result, QueueFuturePoll::Barrier) {
                        QueueFuturePoll::Ready {
                            next_queue: r1_queue,
                            output: r1_ret,
                        } => (r1_queue, r1_ret),
                        _ => unreachable!(),
                    };
                let (r2_queue_mask, r2_ret) =
                    match std::mem::replace(this.inner2_result, QueueFuturePoll::Barrier) {
                        QueueFuturePoll::Ready {
                            next_queue: r2_queue,
                            output: r2_ret,
                        } => (r2_queue, r2_ret),
                        _ => unreachable!(),
                    };
                *this.results_taken = true;
                return QueueFuturePoll::Ready {
                    next_queue: r1_queue_mask.merge(&r2_queue_mask),
                    output: (r1_ret, r2_ret),
                };
            }
        }
    }

    fn dispose(self: Pin<&mut Self>) -> impl std::future::Future<Output = ()> {
        use futures_util::FutureExt;
        let this = self.project();
        futures_util::future::join(this.inner1.dispose(), this.inner2.dispose()).map(|_| ())
    }
}

*/
#[pin_project]
pub struct RunCommandsQueueFuture<I: GPUCommandFuture> {
    #[pin]
    inner: I,

    /// Specified queue to run on.
    queue: QueueRef, // If null, use the previous queue.
    /// Retained state and the timeline index
    retained_state: Option<I::RetainedState>,
    prev_queue: QueueMask,
    output: Option<I::Output>,
}
impl<I: GPUCommandFuture> RunCommandsQueueFuture<I> {
    pub fn new(future: I, queue: QueueRef) -> Self {
        Self {
            inner: future,
            queue,
            retained_state: None,
            prev_queue: QueueMask::empty(),
            output: None,
        }
    }
}
impl<I: GPUCommandFuture> QueueFuture for RunCommandsQueueFuture<I> {
    type Output = I::Output;
    type RecycledState = I::RecycledState;
    type RetainedState = I::RetainedState;

    fn setup(
        self: Pin<&mut Self>,
        ctx: &mut SubmissionContext,
        recycled_state: &mut Self::RecycledState,
        prev_queue: QueueMask,
    ) {
        let this = self.project();
        if this.queue.is_null() {
            let mut iter = prev_queue.iter();
            if let Some(inherited_queue) = iter.next() {
                *this.queue = inherited_queue;
                assert!(
                    iter.next().is_none(),
                    "Cannot use derived queue when the future depends on more than one queues"
                );
            } else {
                // Default to the first queue, if the queue does not have predecessor.
                *this.queue = QueueRef(0);
            }
        }
        *this.prev_queue = prev_queue;
        let mut r = &mut ctx.submission[this.queue.0 as usize];
        let mut temp_command_buffers = Vec::new();
        let mut temp_recording_command_buffers = None;
        let (command_buffers, recording_command_buffer) = match &mut r {
            QueueSubmissionType::Submit {
                command_buffers,
                recording_command_buffer,
            } => (command_buffers, recording_command_buffer),
            _ => (
                &mut temp_command_buffers,
                &mut temp_recording_command_buffers,
            ),
        };
        let q = &mut ctx.queues[this.queue.0 as usize];
        let mut command_ctx = CommandBufferRecordContext {
            stage_index: q.stage_index,
            command_buffers,
            command_pool: ctx.shared_command_pools[q.queue_family_index as usize]
                .as_mut()
                .unwrap(),
            recording_command_buffer,
            timeline_index: q.timeline_index,
            queue: *this.queue,
        };
        match this.inner.init(&mut command_ctx, recycled_state) {
            Some((out, retain)) => {
                *this.output = Some(out);
                *this.retained_state = Some(retain);
            }
            None => (),
        }
        assert!(temp_command_buffers.is_empty());
        assert!(temp_recording_command_buffers.is_none());
    }

    fn record(
        self: Pin<&mut Self>,
        ctx: &mut SubmissionContext,
        recycled_state: &mut Self::RecycledState,
    ) -> QueueFuturePoll<Self::Output> {
        let this = self.project();
        if let Some(output) = this.output.take() {
            return QueueFuturePoll::Ready {
                next_queue: {
                    let mut mask = QueueMask::empty();
                    mask.set_queue(*this.queue);
                    mask
                },
                output,
            };
        }

        let queue = {
            let mut mask = QueueMask::empty();
            mask.set_queue(*this.queue);
            mask
        };

        if !this.prev_queue.is_empty() {
            let ret = if *this.prev_queue == queue {
                QueueFuturePoll::Barrier
            } else {
                QueueFuturePoll::Semaphore(Vec::new())
            };
            *this.prev_queue = QueueMask::empty();
            return ret;
        }

        let mut r = &mut ctx.submission[this.queue.0 as usize];

        match &r {
            QueueSubmissionType::Unknown => {
                *r = QueueSubmissionType::Submit {
                    command_buffers: Vec::new(),
                    recording_command_buffer: None,
                };
            }
            QueueSubmissionType::Submit { .. } => (),
            _ => panic!(),
        };

        let (command_buffers, recording_command_buffer) = match &mut r {
            QueueSubmissionType::Submit {
                command_buffers,
                recording_command_buffer,
            } => (command_buffers, recording_command_buffer),
            _ => unreachable!(),
        };
        let q = &mut ctx.queues[this.queue.0 as usize];
        let mut command_ctx = CommandBufferRecordContext {
            stage_index: q.stage_index,
            command_buffers,
            command_pool: ctx.shared_command_pools[q.queue_family_index as usize]
                .as_mut()
                .unwrap(),
            recording_command_buffer,
            timeline_index: q.timeline_index,
            queue: *this.queue,
        };

        let poll = command_ctx.record_one_step(this.inner, recycled_state, |a| {
            for (img, barrier) in a.image_accesses.iter() {
                assert!(
                    barrier.src_layout != barrier.dst_layout
                        || barrier.src_queue_family != barrier.dst_queue_family
                );
                if barrier.src_queue.is_null() {
                    assert!(barrier.src_layout != barrier.dst_layout);
                    continue;
                }
                ctx.queues[barrier.src_queue.0 as usize].exports.push(
                    QueueSubmissionContextExport::Image {
                        image: img.clone(),
                        barrier: barrier.barrier,
                        dst_queue_family: barrier.dst_queue_family,
                        src_layout: barrier.src_layout,
                        dst_layout: barrier.dst_layout,
                    },
                );
            }
            for (buffer, barrier) in a.buffer_accesses.iter() {
                assert!(barrier.src_queue_family != barrier.dst_queue_family);
                ctx.queues[barrier.src_queue.0 as usize].exports.push(
                    QueueSubmissionContextExport::Buffer {
                        buffer: buffer.clone(),
                        barrier: barrier.barrier,
                        dst_queue_family: barrier.dst_queue_family,
                    },
                );
            }
            for transition in &a.semaphore_transitions {
                match transition {
                    StageContextSemaphoreTransition::Managed {
                        src_queue,
                        dst_queue,
                        src_stages,
                        dst_stages,
                    } => {
                        ctx.queues[src_queue.0 as usize]
                            .signals
                            .insert((*src_stages, false));
                        ctx.queues[dst_queue.0 as usize].waits.push(
                            QueueSubmissionContextSemaphoreWait::WaitForSignal {
                                dst_stages: *dst_stages,
                                queue: *src_queue,
                                src_stages: *src_stages,
                            },
                        );
                    }
                    StageContextSemaphoreTransition::Untracked {
                        semaphore,
                        dst_queue,
                        dst_stages,
                    } => {
                        ctx.queues[dst_queue.0 as usize].waits.push(
                            QueueSubmissionContextSemaphoreWait::WaitForAcquire {
                                dst_stages: *dst_stages,
                                acquire_semaphore: *semaphore,
                            },
                        );
                    }
                }
            }
        });
        let result = match poll {
            Poll::Ready((output, retained_state)) => {
                *this.retained_state = Some(retained_state);
                QueueFuturePoll::Ready {
                    next_queue: {
                        let mut mask = QueueMask::empty();
                        mask.set_queue(*this.queue);
                        mask
                    },
                    output,
                }
            }
            Poll::Pending => QueueFuturePoll::Barrier,
        };
        let r = &mut ctx.queues[this.queue.0 as usize];
        r.stage_index += 1;
        result
    }

    fn dispose(mut self) -> Self::RetainedState {
        self.retained_state.take().unwrap()
    }
}
