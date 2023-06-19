use crate::{QueueRef, RunCommandsQueueFuture};

use super::{CommandBufferRecordContext, Disposable, GPUCommandFuture, StageContext};
use ash::vk;
use pin_project::pin_project;
use std::any::Any;
use std::cell::RefCell;
use std::marker::PhantomData;

use std::pin::Pin;
use std::task::Poll;

pub trait GPUCommandFutureExt: GPUCommandFuture + Sized {
    fn join<G: GPUCommandFuture>(self, other: G) -> GPUCommandJoin<Self, G> {
        GPUCommandJoin {
            inner1: self,
            inner1_result: None,
            inner2: other,
            inner2_result: None,
            results_taken: false,
        }
    }
    fn map<R, F: FnOnce(Self::Output) -> R>(self, mapper: F) -> GPUCommandMap<Self, F> {
        GPUCommandMap {
            inner: self,
            mapper: Some(mapper),
        }
    }
    fn schedule_on_queue(self, queue: QueueRef) -> RunCommandsQueueFuture<Self> {
        RunCommandsQueueFuture::new(self, queue)
    }
    fn schedule(self) -> RunCommandsQueueFuture<Self> {
        RunCommandsQueueFuture::new(self, QueueRef::null())
    }
    fn dynamic(self) -> DynamicCommandFuture<Self::Output>
    where
        Self::RetainedState: 'static,
        Self::RecycledState: 'static,
        Self: 'static + Send + Sync,
    {
        DynamicCommandFuture {
            inner: Box::pin(self),
            _marker: PhantomData,
        }
    }
}

impl<T: GPUCommandFuture> GPUCommandFutureExt for T {}

#[pin_project]
pub struct GPUCommandJoin<G1, G2>
where
    G1: GPUCommandFuture,
    G2: GPUCommandFuture,
{
    #[pin]
    inner1: G1,
    inner1_result: Option<(G1::Output, G1::RetainedState)>,
    #[pin]
    inner2: G2,
    inner2_result: Option<(G2::Output, G2::RetainedState)>,

    results_taken: bool,
}

impl<G1, G2> GPUCommandFuture for GPUCommandJoin<G1, G2>
where
    G1: GPUCommandFuture,
    G2: GPUCommandFuture,
{
    type Output = (G1::Output, G2::Output);
    type RetainedState = (G1::RetainedState, G2::RetainedState);
    type RecycledState = (G1::RecycledState, G2::RecycledState);
    #[inline]
    fn record(
        self: Pin<&mut Self>,
        command_buffer: &mut CommandBufferRecordContext,
        recycled_state: &mut Self::RecycledState,
    ) -> Poll<(Self::Output, Self::RetainedState)> {
        let this = self.project();
        assert!(
            !*this.results_taken,
            "Attempted to record a GPUCommandJoin after it's finished"
        );
        let (recycle1, recycle2) = recycled_state;
        if this.inner1_result.is_none() {
            if let Poll::Ready(r) = this.inner1.record(command_buffer, recycle1) {
                *this.inner1_result = Some(r);
            }
        }
        if this.inner2_result.is_none() {
            if let Poll::Ready(r) = this.inner2.record(command_buffer, recycle2) {
                *this.inner2_result = Some(r);
            }
        }
        if this.inner1_result.is_some() && this.inner2_result.is_some() {
            let (r1_ret, r1_retained_state) = this.inner1_result.take().unwrap();
            let (r2_ret, r2_retained_state) = this.inner2_result.take().unwrap();
            *this.results_taken = true;
            Poll::Ready(((r1_ret, r2_ret), (r1_retained_state, r2_retained_state)))
        } else {
            Poll::Pending
        }
    }

    fn context(self: Pin<&mut Self>, ctx: &mut StageContext) {
        let this = self.project();
        assert!(
            !*this.results_taken,
            "Attempted to take the context of a GPUCommandJoin after it's finished"
        );
        if this.inner1_result.is_none() {
            this.inner1.context(ctx);
        }
        if this.inner2_result.is_none() {
            this.inner2.context(ctx);
        }
    }

    fn init(
        self: Pin<&mut Self>,
        ctx: &mut CommandBufferRecordContext,
        recycled_state: &mut Self::RecycledState,
    ) -> Option<(Self::Output, Self::RetainedState)> {
        let this = self.project();
        if let Some(r) = this.inner1.init(ctx, &mut recycled_state.0) {
            *this.inner1_result = Some(r);
        }
        if let Some(r) = this.inner2.init(ctx, &mut recycled_state.1) {
            *this.inner2_result = Some(r);
        }
        if this.inner1_result.is_some() && this.inner2_result.is_some() {
            let (r1_ret, r1_retained_state) = this.inner1_result.take().unwrap();
            let (r2_ret, r2_retained_state) = this.inner2_result.take().unwrap();
            *this.results_taken = true;
            Some(((r1_ret, r2_ret), (r1_retained_state, r2_retained_state)))
        } else {
            None
        }
    }
}

#[pin_project]
pub struct GPUCommandJoinVec<F>
where
    F: GPUCommandFuture,
{
    #[pin]
    inner: Box<[F]>,
    results: Vec<Option<(F::Output, F::RetainedState)>>,
    results_taken: bool,
}

impl<F> GPUCommandFuture for GPUCommandJoinVec<F>
where
    F: GPUCommandFuture,
{
    type Output = Vec<F::Output>;
    type RetainedState = Vec<F::RetainedState>;
    type RecycledState = Vec<F::RecycledState>;
    #[inline]
    fn record(
        self: Pin<&mut Self>,
        command_buffer: &mut CommandBufferRecordContext,
        recycled_state: &mut Self::RecycledState,
    ) -> Poll<(Self::Output, Self::RetainedState)> {
        let mut this = self.project();
        assert_eq!(recycled_state.len(), this.inner.len());
        assert!(
            !*this.results_taken,
            "Attempted to record a GPUCommandJoin after it's finished"
        );
        for ((future, result), recycled_state) in this
            .inner
            .iter_mut()
            .zip(this.results.iter_mut())
            .zip(recycled_state.iter_mut())
        {
            if result.is_none() {
                let future = unsafe { Pin::new_unchecked(future) };
                if let Poll::Ready(r) = future.record(command_buffer, recycled_state) {
                    *result = Some(r);
                }
            }
        }
        if this.results.iter().all(|a| a.is_some()) {
            let (results, retained_states): (Vec<_>, Vec<_>) =
                this.results.drain(..).map(|a| a.unwrap()).unzip();
            *this.results_taken = true;
            Poll::Ready((results, retained_states))
        } else {
            Poll::Pending
        }
    }

    fn context(self: Pin<&mut Self>, ctx: &mut StageContext) {
        let mut this = self.project();
        assert!(
            !*this.results_taken,
            "Attempted to take the context of a GPUCommandJoin after it's finished"
        );
        for (result, future) in this.results.iter().zip(this.inner.iter_mut()) {
            if result.is_none() {
                let future = unsafe { Pin::new_unchecked(future) };
                future.context(ctx);
            }
        }
    }

    fn init(
        self: Pin<&mut Self>,
        ctx: &mut CommandBufferRecordContext,
        recycled_state: &mut Self::RecycledState,
    ) -> Option<(Self::Output, Self::RetainedState)> {
        let mut this = self.project();
        if recycled_state.len() == 0 {
            *recycled_state = this.inner.iter().map(|_| Default::default()).collect();
            // TODO: optimize for zero-sized objs
        }
        assert_eq!(recycled_state.len(), this.inner.len());

        for ((result, future), recycled_state) in this
            .results
            .iter_mut()
            .zip(this.inner.iter_mut())
            .zip(recycled_state)
        {
            let future = unsafe { Pin::new_unchecked(future) };
            if let Some(r) = future.init(ctx, recycled_state) {
                *result = Some(r);
            }
        }

        if this.results.iter().all(|a| a.is_some()) {
            let (results, retained_states): (Vec<_>, Vec<_>) =
                this.results.drain(..).map(|a| a.unwrap()).unzip();
            *this.results_taken = true;
            Some((results, retained_states))
        } else {
            None
        }
    }
}

#[pin_project]
pub struct GPUCommandMap<G, F> {
    mapper: Option<F>,
    #[pin]
    inner: G,
}

impl<G, R, F> GPUCommandFuture for GPUCommandMap<G, F>
where
    G: GPUCommandFuture,
    F: FnOnce(G::Output) -> R,
{
    type Output = R;
    type RetainedState = G::RetainedState;
    type RecycledState = G::RecycledState;
    #[inline]
    fn record(
        self: Pin<&mut Self>,
        ctx: &mut CommandBufferRecordContext,
        recycled_state: &mut Self::RecycledState,
    ) -> Poll<(Self::Output, Self::RetainedState)> {
        let this = self.project();
        match this.inner.record(ctx, recycled_state) {
            Poll::Pending => Poll::Pending,
            Poll::Ready((r, retained_state)) => {
                let mapper = this
                    .mapper
                    .take()
                    .expect("Attempted to poll GPUCommandMap after completion");
                Poll::Ready(((mapper)(r), retained_state))
            }
        }
    }
    fn context(self: Pin<&mut Self>, ctx: &mut StageContext) {
        self.project().inner.context(ctx);
    }
    fn init(
        self: Pin<&mut Self>,
        ctx: &mut CommandBufferRecordContext,
        recycled_state: &mut Self::RecycledState,
    ) -> Option<(Self::Output, Self::RetainedState)> {
        let this = self.project();
        this.inner.init(ctx, recycled_state).map(|(out, retain)| {
            let mapper = this.mapper.take().unwrap();
            ((mapper)(out), retain)
        })
    }
}

#[pin_project(project = GPUCommandForkedStateInnerProj)]
pub enum GPUCommandForkedStateInner<G: GPUCommandFuture> {
    Some(#[pin] G),
    Resolved(G::Output),
}
impl<G: GPUCommandFuture> GPUCommandForkedStateInner<G> {
    pub fn unwrap_pinned(self: Pin<&mut Self>) -> Pin<&mut G> {
        use GPUCommandForkedStateInnerProj::*;
        match self.project() {
            Some(g) => g,
            Resolved(_) => panic!(),
        }
    }
}

// The shared structure between all branches
pub struct GPUCommandForkedInner<'a, G: GPUCommandFuture, const N: usize> {
    inner: Pin<&'a mut GPUCommandForkedStateInner<G>>,
    last_stage: u32,
    ready: [bool; N],
}
impl<'a, G: GPUCommandFuture, const N: usize> GPUCommandForkedInner<'a, G, N> {
    pub fn new(inner: Pin<&'a mut GPUCommandForkedStateInner<G>>) -> RefCell<Self> {
        RefCell::new(Self {
            inner,
            last_stage: 0,
            ready: [false; N],
        })
    }
}

/// The structure specific to a single branch
#[pin_project]
pub struct GPUCommandForked<'a, 'r, G: GPUCommandFuture, const N: usize> {
    inner: &'r RefCell<GPUCommandForkedInner<'a, G, N>>,
    id: usize,
}
impl<'a, 'r, G: GPUCommandFuture, const N: usize> GPUCommandForked<'a, 'r, G, N> {
    pub fn new(inner: &'r RefCell<GPUCommandForkedInner<'a, G, N>>, id: usize) -> Self {
        Self { inner, id }
    }
}

impl<'a, 'r, G, const N: usize> GPUCommandFuture for GPUCommandForked<'a, 'r, G, N>
where
    G: GPUCommandFuture,
    G::Output: Clone,
{
    type Output = G::Output;
    type RetainedState = Option<G::RetainedState>;
    type RecycledState = G::RecycledState;
    #[inline]
    fn record(
        self: Pin<&mut Self>,
        ctx: &mut CommandBufferRecordContext,
        recycled_state: &mut Self::RecycledState,
    ) -> Poll<(Self::Output, Self::RetainedState)> {
        let this = &mut *self.project().inner.borrow_mut();
        if !this.ready.iter().all(|a| *a) {
            // no op.
            return Poll::Pending;
        }
        match this.inner.as_mut().project() {
            GPUCommandForkedStateInnerProj::Resolved(result) => Poll::Ready((result.clone(), None)),
            GPUCommandForkedStateInnerProj::Some(inner) => {
                if this.last_stage < ctx.current_stage_index() {
                    // do the work
                    this.last_stage = ctx.current_stage_index();
                    match inner.record(ctx, recycled_state) {
                        Poll::Pending => Poll::Pending,
                        Poll::Ready((result, retained)) => {
                            this.inner
                                .set(GPUCommandForkedStateInner::Resolved(result.clone()));
                            Poll::Ready((result, Some(retained)))
                        }
                    }
                } else {
                    Poll::Pending
                }
            }
        }
    }
    fn context(self: Pin<&mut Self>, ctx: &mut StageContext) {
        let id = self.id;
        let mut this = self.project().inner.borrow_mut();
        this.ready[id] = true;
        if !this.ready.iter().all(|a| *a) {
            // no op. only the one that finishes the latest will actually record and generate dependencies.
            return;
        }
        this.inner.as_mut().unwrap_pinned().context(ctx);
    }
    fn init(
        self: Pin<&mut Self>,
        _ctx: &mut CommandBufferRecordContext,
        _recycled_state: &mut Self::RecycledState,
    ) -> Option<(Self::Output, Self::RetainedState)> {
        // Noop. The inner command will be initialized when fork was called on it.
        None
    }
}

#[pin_project]
pub struct DynamicCommandFuture<Out> {
    inner: Pin<Box<dyn DynamicCommandFutureTrait<Output = Out>>>,
    _marker: PhantomData<Out>,
}

trait DynamicCommandFutureTrait: Send + Sync {
    type Output;
    fn record(
        self: Pin<&mut Self>,
        ctx: &mut CommandBufferRecordContext,
        recycled_state: &mut Option<Box<dyn Any + Send + Sync>>,
    ) -> Poll<(Self::Output, Box<dyn Disposable + Send>)>;
    fn context(self: Pin<&mut Self>, ctx: &mut StageContext);
    fn init(
        self: Pin<&mut Self>,
        _ctx: &mut CommandBufferRecordContext,
        _recycled_state: &mut Option<Box<dyn Any + Send + Sync>>,
    ) -> Option<(Self::Output, Box<dyn Disposable + Send>)>;
}
impl<F: GPUCommandFuture> DynamicCommandFutureTrait for F
where
    F::RecycledState: 'static,
    F::RetainedState: 'static,
    F: Send + Sync,
{
    type Output = F::Output;
    fn record(
        self: Pin<&mut Self>,
        ctx: &mut CommandBufferRecordContext,
        recycled_state: &mut Option<Box<dyn Any + Send + Sync>>,
    ) -> Poll<(Self::Output, Box<dyn Disposable + Send>)> {
        if recycled_state.is_none() {
            let def = F::RecycledState::default();
            *recycled_state = Some(Box::new(def));
        }
        let recycled_state: &mut F::RecycledState =
            recycled_state.as_mut().unwrap().downcast_mut().unwrap();

        let poll = <Self as GPUCommandFuture>::record(self, ctx, recycled_state);
        match poll {
            Poll::Pending => Poll::Pending,
            Poll::Ready((output, retained_state)) => {
                Poll::Ready((output, Box::new(retained_state)))
            }
        }
    }
    fn context(self: Pin<&mut Self>, ctx: &mut StageContext) {
        <Self as GPUCommandFuture>::context(self, ctx)
    }
    fn init(
        self: Pin<&mut Self>,
        ctx: &mut CommandBufferRecordContext,
        recycled_state: &mut Option<Box<dyn Any + Send + Sync>>,
    ) -> Option<(Self::Output, Box<dyn Disposable + Send>)> {
        if recycled_state.is_none() {
            let def = F::RecycledState::default();
            *recycled_state = Some(Box::new(def));
        }
        let recycled_state: &mut F::RecycledState =
            recycled_state.as_mut().unwrap().downcast_mut().unwrap();
        let poll = <Self as GPUCommandFuture>::init(self, ctx, recycled_state);
        match poll {
            None => None,
            Some((output, retained_state)) => Some((output, Box::new(retained_state))),
        }
    }
}

impl<OUT> GPUCommandFuture for DynamicCommandFuture<OUT> {
    type Output = OUT;

    type RetainedState = Box<dyn Disposable + Send>;

    type RecycledState = Option<Box<dyn Any + Send + Sync>>;

    fn record(
        self: Pin<&mut Self>,
        ctx: &mut CommandBufferRecordContext,
        recycled_state: &mut Self::RecycledState,
    ) -> Poll<(Self::Output, Self::RetainedState)> {
        let this = self.project();
        this.inner.as_mut().record(ctx, recycled_state)
    }

    fn context(self: Pin<&mut Self>, ctx: &mut StageContext) {
        let this = self.project();
        this.inner.as_mut().context(ctx);
    }
    fn init(
        self: Pin<&mut Self>,
        ctx: &mut CommandBufferRecordContext,
        recycled_state: &mut Self::RecycledState,
    ) -> Option<(Self::Output, Self::RetainedState)> {
        let this = self.project();
        this.inner.as_mut().init(ctx, recycled_state)
    }
}

pub fn join_vec<T: GPUCommandFuture>(vec: Vec<T>) -> GPUCommandJoinVec<T> {
    let results = vec.iter().map(|_| None).collect();
    GPUCommandJoinVec {
        inner: vec.into_boxed_slice(),
        results,
        results_taken: false,
    }
}

pub fn run<Cmd, Ctx>(command: Cmd, ctx: Ctx) -> InlineCommandFuture<Cmd, Ctx>
where
    Ctx: FnOnce(&mut rhyolite::future::StageContext),
    Cmd: FnOnce(&rhyolite::future::CommandBufferRecordContext, vk::CommandBuffer),
{
    InlineCommandFuture {
        command: Some(command),
        ctx: Some(ctx),
    }
}
// TODO: group base alignment
#[pin_project]
pub struct InlineCommandFuture<Cmd, Ctx>
where
    Ctx: FnOnce(&mut rhyolite::future::StageContext),
    Cmd: FnOnce(&rhyolite::future::CommandBufferRecordContext, vk::CommandBuffer),
{
    command: Option<Cmd>,
    ctx: Option<Ctx>,
}

impl<Cmd, Ctx> GPUCommandFuture for InlineCommandFuture<Cmd, Ctx>
where
    Ctx: FnOnce(&mut rhyolite::future::StageContext),
    Cmd: FnOnce(&rhyolite::future::CommandBufferRecordContext, vk::CommandBuffer),
{
    type Output = ();
    type RetainedState = ();

    type RecycledState = ();

    fn record(
        mut self: std::pin::Pin<&mut Self>,
        ctx: &mut rhyolite::future::CommandBufferRecordContext,
        recycled_state: &mut Self::RecycledState,
    ) -> std::task::Poll<(Self::Output, Self::RetainedState)> {
        ctx.record(self.command.take().unwrap());
        std::task::Poll::Ready(((), ()))
    }

    fn context(mut self: std::pin::Pin<&mut Self>, ctx: &mut rhyolite::future::StageContext) {
        (self.ctx.take().unwrap())(ctx);
    }
}
