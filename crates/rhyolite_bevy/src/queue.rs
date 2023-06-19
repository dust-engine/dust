use std::{
    cell::RefCell,
    future::Future,
    ops::{Deref, DerefMut},
    sync::{atomic::AtomicU64, Arc},
};

use bevy_ecs::system::{ResMut, Resource};
use crossbeam_channel::{Receiver, Sender};
use rhyolite::QueueCompileExt;
use rhyolite::{
    ash::vk,
    commands::SharedCommandPool,
    future::{use_per_frame_state_blocking, Disposable, PerFrameContainer, PerFrameState},
    utils::retainer::Retainer,
    CachedStageSubmissions, FencePool, FencePoolLike, HasDevice, QueueFuture,
    TimelineSemaphorePool,
};
use thread_local::ThreadLocal;

pub struct Frame {
    shared_command_pools: Vec<Option<SharedCommandPool>>,
    pub(crate) shared_semaphore_pool: TimelineSemaphorePool,
    shared_fence_pool: FencePool,
}

struct AsyncBatch {
    pub(crate) shared_semaphore_pool: TimelineSemaphorePool,
    pub(crate) shared_command_pool: Vec<Option<SharedCommandPool>>,
    generation: u64,
}

pub struct AsyncQueuesInner {
    current_batch: ThreadLocal<RefCell<Retainer<PerFrameContainer<AsyncBatch>>>>,
    batch_queue: PerFrameState<AsyncBatch>,
    generation: AtomicU64,
}

pub struct CompiledQueueFutureErased {
    pub(super) submission_batch: Vec<CachedStageSubmissions>,
    event: event_listener::Event,
}

#[derive(Clone, Resource)]
pub struct AsyncQueues {
    device: Arc<rhyolite::Device>,
    inner: Arc<AsyncQueuesInner>,
    sender: Sender<CompiledQueueFutureErased>,
}

impl AsyncQueues {
    fn current_batch<R>(
        &self,
        callback: impl FnOnce(&mut Retainer<PerFrameContainer<AsyncBatch>>) -> R,
    ) -> R {
        let create_batch = || AsyncBatch {
            generation: self
                .inner
                .generation
                .load(std::sync::atomic::Ordering::Relaxed),
            shared_semaphore_pool: TimelineSemaphorePool::new(self.device.clone()),
            shared_command_pool: make_shared_command_pools(self.device.clone()),
        };

        let batch = self.inner.current_batch.get_or(|| {
            let a = self.inner.batch_queue.use_state(create_batch).reuse(|a| {
                for pool in a.shared_command_pool.iter_mut().filter_map(|a| a.as_mut()) {
                    pool.reset(true);
                }
                a.shared_semaphore_pool.reset();
            });
            RefCell::new(Retainer::new(a))
        });
        let mut b = batch.borrow_mut();
        let retainer: &mut Retainer<PerFrameContainer<AsyncBatch>> = b.deref_mut();
        if retainer.generation
            < self
                .inner
                .generation
                .load(std::sync::atomic::Ordering::Relaxed)
        {
            std::mem::replace(
                retainer,
                Retainer::new(self.inner.batch_queue.use_state(create_batch)),
            );
        }
        callback(retainer)
    }
    pub fn submit<F: QueueFuture>(
        &self,
        future: F,
        recycled_state: &mut F::RecycledState,
    ) -> impl Future<Output = F::Output> + Send
    where
        F::Output: 'static + Send,
        F::RetainedState: 'static + Send,
    {
        self.current_batch(|batch| {
            let guard = batch.handle();
            let batch: &mut AsyncBatch = batch.deref_mut();
            let compiled = future.compile(
                &mut batch.shared_command_pool,
                &mut batch.shared_semaphore_pool,
                recycled_state,
                false,
            );
            assert!(compiled.final_signals.is_none());
            let listener = if !compiled.is_empty() {
                let event = event_listener::Event::new();
                let listener = event.listen();
                self.sender
                    .send(CompiledQueueFutureErased {
                        submission_batch: compiled.submission_batch,
                        event,
                    })
                    .unwrap();
                Some(listener)
            } else {
                None
            };

            // By using a retainer to wrap around PerFrameContainer and drop the handle only after
            // awaiting on the future, we ensure that old frames aren't getting reused by newer frames
            // before the old frame actually finishes rendering.
            let out = compiled.output;
            let dispose = compiled.fut_dispose;
            async {
                if let Some(listener) = listener {
                    listener.await;
                }
                dispose.dispose();
                drop(guard);
                out
            }
        })
    }
}

struct AsyncFenceRecycler {
    device: Arc<rhyolite::Device>,
    fences: Receiver<vk::Fence>,
}
impl FencePoolLike for AsyncFenceRecycler {
    fn get(&mut self) -> vk::Fence {
        self.fences
            .try_recv()
            .ok()
            .map(|fence| unsafe {
                self.device.reset_fences(&[fence]).unwrap();
                fence
            })
            .unwrap_or_else(|| unsafe {
                self.device
                    .create_fence(&vk::FenceCreateInfo::default(), None)
                    .unwrap()
            })
    }
}
impl Drop for AsyncFenceRecycler {
    fn drop(&mut self) {
        while let Some(fence) = self.fences.try_recv().ok() {
            unsafe {
                self.device.destroy_fence(fence, None);
            }
        }
    }
}

#[derive(Resource)]
pub struct Queues {
    queues: rhyolite::Queues,
    frames: PerFrameState<Frame>,
    current_frame: Option<Retainer<PerFrameContainer<Frame>>>,
    max_frame_in_flight: usize,
    task_pool: bevy_tasks::TaskPool,

    async_submission_receiver: Receiver<CompiledQueueFutureErased>,
    pub async_queues: AsyncQueues,

    fence_recycler: AsyncFenceRecycler,
    fence_recycler_sender: Sender<vk::Fence>,
}

/// Safety: async_submission_receiver should never be used when we only have &self
unsafe impl Sync for Queues {}
impl HasDevice for Queues {
    fn device(&self) -> &Arc<rhyolite::Device> {
        self.queues.device()
    }
}

impl Queues {
    pub fn new(queues: rhyolite::Queues, max_frame_in_flight: usize) -> Self {
        let device = queues.device().clone();
        let (sender, receiver) = crossbeam_channel::unbounded();
        let async_queues = AsyncQueues {
            device: device.clone(),
            inner: Arc::new(AsyncQueuesInner {
                current_batch: ThreadLocal::new(),
                batch_queue: PerFrameState::default(),
                generation: AtomicU64::new(0),
            }),
            sender,
        };
        let (fence_recycler_sender, fence_recycler_receiver) = crossbeam_channel::unbounded();
        Self {
            queues,
            frames: Default::default(),
            current_frame: None,
            max_frame_in_flight,
            task_pool: bevy_tasks::TaskPool::new(),
            async_submission_receiver: receiver,
            async_queues,
            fence_recycler: AsyncFenceRecycler {
                device,
                fences: fence_recycler_receiver,
            },
            fence_recycler_sender,
        }
    }
    /// May block. Runs in SetUp stage.
    pub fn next_frame(&mut self) {
        let frame =
            use_per_frame_state_blocking(&mut self.frames, self.max_frame_in_flight, || Frame {
                shared_command_pools: make_shared_command_pools(self.queues.device().clone()),
                shared_fence_pool: rhyolite::FencePool::new(self.queues.device().clone()),
                shared_semaphore_pool: rhyolite::TimelineSemaphorePool::new(
                    self.queues.device().clone(),
                ),
            })
            .reuse(|frame| {
                for i in frame.shared_command_pools.iter_mut() {
                    if let Some(i) = i {
                        i.reset(false);
                    }
                }
                frame.shared_fence_pool.reset();
                frame.shared_semaphore_pool.reset();
            });
        self.current_frame = Some(Retainer::new(frame));
    }
    /// Runs in CleanUp stage
    pub fn flush(&mut self) {
        while let Some(submission) = self.async_submission_receiver.try_recv().ok() {
            unsafe {
                let fences = self.queues.submit_compiled(
                    submission.submission_batch.into_iter(),
                    &mut self.fence_recycler,
                );
                let device = self.queues.device().clone();
                let sender = self.fence_recycler_sender.clone();
                blocking::unblock(move || {
                    device.wait_for_fences(&fences, true, !0).unwrap();
                    for fence in fences.into_iter() {
                        sender.send(fence).unwrap();
                    }
                    submission.event.notify(1);
                })
                .detach();
            }
        }
    }
    pub fn current_frame(&mut self) -> &mut Frame {
        self.current_frame.as_mut().unwrap().deref_mut()
    }
    pub fn num_frame_in_flight(&self) -> u32 {
        self.max_frame_in_flight as u32
    }
    pub fn submit<F: QueueFuture<Output = ()>>(
        &mut self,
        future: F,
        recycled_state: &mut F::RecycledState,
    ) where
        F::Output: 'static,
        F::RetainedState: 'static,
    {
        let current_frame: &mut Frame = self.current_frame.as_mut().unwrap().deref_mut();
        let future = self.queues.submit(
            future,
            &mut current_frame.shared_command_pools,
            &mut current_frame.shared_semaphore_pool,
            &mut current_frame.shared_fence_pool,
            recycled_state,
            // No need to signal final semaphores, because we're always going to detach the task and wait for it on the host.
            false,
        );

        // By using a retainer to wrap around PerFrameContainer and drop the handle only after
        // awaiting on the future, we ensure that old frames aren't getting reused by newer frames
        // before the old frame actually finishes rendering.
        let guard = self.current_frame.as_ref().unwrap().handle();
        let task = async {
            let out = future.await;
            drop(guard);
            out
        };
        self.task_pool.spawn(task).detach();
    }
}

#[derive(Resource)]
pub struct QueuesRouter(rhyolite::QueuesRouter);
impl QueuesRouter {
    pub fn new(inner: rhyolite::QueuesRouter) -> Self {
        Self(inner)
    }
}

impl Deref for QueuesRouter {
    type Target = rhyolite::QueuesRouter;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Create SharedCommandPool for all queue family with at least one queue
pub fn make_shared_command_pools(device: Arc<rhyolite::Device>) -> Vec<Option<SharedCommandPool>> {
    device
        .queue_info()
        .families
        .iter()
        .enumerate()
        .map(|(queue_family_index, queues)| {
            if queues.is_empty() {
                None
            } else {
                Some(SharedCommandPool::new(
                    device.clone(),
                    queue_family_index as u32,
                ))
            }
        })
        .collect()
}

pub fn flush_async_queue_system(mut queue: ResMut<Queues>) {
    queue.flush();
}
