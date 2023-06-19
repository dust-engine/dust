use super::{CommandBufferRecordContext, Disposable, GPUCommandFuture, StageContext};
use pin_project::pin_project;
use std::marker::PhantomData;
use std::ops::{Generator, GeneratorState};
use std::pin::Pin;
use std::task::Poll;

pub struct GPUCommandGeneratorContextFetchPtr {
    this: *mut (),
    fetch: fn(*mut (), ctx: &mut StageContext),
}
impl GPUCommandGeneratorContextFetchPtr {
    pub fn new<T: GPUCommandFuture>(this: Pin<&mut T>) -> Self {
        Self {
            this: unsafe { this.get_unchecked_mut() as *mut T as *mut () },
            fetch: |ptr, stage| unsafe {
                let ptr = std::pin::Pin::new_unchecked(&mut *(ptr as *mut T));
                T::context(ptr, stage)
            },
        }
    }
    pub fn call(&mut self, ctx: &mut StageContext) {
        (self.fetch)(self.this, ctx);
    }
}
//S Generator takes a raw pointer as the argument. https://github.com/rust-lang/rust/issues/68923
// Should be:
// pub trait GPUCommandGenerator<'retain, R, State, Recycle: Default> = for<'a, 'b> Generator<
// (&'a mut CommandBufferRecordContext<'b>, &'a mut Recycle),
// Yield = GPUCommandGeneratorContextFetchPtr,
// Return = (R, State, PhantomData<&'retain ()>),
// >;
pub trait GPUCommandGenerator<R, State, Recycle: Default> = Generator<
    (*mut (), *mut Recycle),
    Yield = GPUCommandGeneratorContextFetchPtr,
    Return = (R, State),
>;

enum GPUCommandBlockState {
    /// The initial state. After init() was called, the future is guaranteed not to be in this state.
    Initial,
    /// The "normal" state. Call the attached `next_ctx` to fetch the context for the next stage.
    Continue {
        next_ctx: GPUCommandGeneratorContextFetchPtr,
    },
    /// Only occurs when the generator returns without yielding anything during init.
    EarlyTerminated,
    Terminated,
}
#[pin_project]
pub struct GPUCommandBlock<R, Retain, Recycle: Default, G> {
    #[pin]
    inner: G,
    state: GPUCommandBlockState,
    _marker: std::marker::PhantomData<fn(*mut Recycle) -> (R, Retain)>,
}

/// TODO: This is a bad workaround. We use raw pointers inside the generator and rust had some problems
/// figuring out their lifetimes, wrongly assuming that they will live across yield points. This causes
/// Rust to mark the inner generator as Send.
unsafe impl<R, State, Recycle: Default, G: GPUCommandGenerator<R, State, Recycle>> Send
    for GPUCommandBlock<R, State, Recycle, G>
{
}

impl<R, State, Recycle: Default, G: GPUCommandGenerator<R, State, Recycle>>
    GPUCommandBlock<R, State, Recycle, G>
{
    pub fn new(inner: G) -> Self {
        Self {
            inner,
            state: GPUCommandBlockState::Initial,
            _marker: PhantomData,
        }
    }
}
impl<
        R,
        State: Disposable + Send,
        Recycle: Default + Send + Sync,
        G: GPUCommandGenerator<R, State, Recycle>,
    > GPUCommandFuture for GPUCommandBlock<R, State, Recycle, G>
{
    type Output = R;
    type RetainedState = State;
    type RecycledState = Recycle;
    fn record(
        self: Pin<&mut Self>,
        ctx: &mut CommandBufferRecordContext,
        recycled_state: &mut Recycle,
    ) -> Poll<(Self::Output, Self::RetainedState)> {
        let this = self.project();
        match this.state {
            GPUCommandBlockState::Initial => panic!("Calling record without calling init"),
            GPUCommandBlockState::Continue { .. } => (),
            GPUCommandBlockState::EarlyTerminated | GPUCommandBlockState::Terminated => {
                panic!("Attempts to call record after ending")
            }
        }

        match this
            .inner
            .resume((ctx as *mut _ as *mut (), recycled_state))
        {
            GeneratorState::Yielded(ctx) => {
                *this.state = GPUCommandBlockState::Continue { next_ctx: ctx };
                // continue here.
                Poll::Pending
            }
            GeneratorState::Complete((ret, state)) => {
                *this.state = GPUCommandBlockState::Terminated;
                Poll::Ready((ret, state))
            }
        }
    }
    fn context(self: Pin<&mut Self>, ctx: &mut StageContext) {
        let this = self.project();
        match this.state {
            GPUCommandBlockState::Initial => panic!("Calling context without calling init"),
            GPUCommandBlockState::Continue { ref mut next_ctx } => next_ctx.call(ctx),
            GPUCommandBlockState::EarlyTerminated | GPUCommandBlockState::Terminated => {
                panic!("Attempts to call context after generator ending")
            }
        }
    }
    fn init(
        self: Pin<&mut Self>,
        ctx: &mut CommandBufferRecordContext,
        recycled_state: &mut Recycle,
    ) -> Option<(Self::Output, Self::RetainedState)> {
        match self.state {
            GPUCommandBlockState::Initial => (),
            _ => unreachable!(),
        };
        let this = self.project();

        // Reach the first yield point to get the context of the first awaited future.
        match this
            .inner
            .resume((ctx as *mut _ as *mut (), recycled_state))
        {
            GeneratorState::Yielded(ctx) => {
                *this.state = GPUCommandBlockState::Continue { next_ctx: ctx };
            }
            GeneratorState::Complete((output, retain)) => {
                // We're pretty sure that this should be the first time we pull the generator.
                // However, it's already completed. This indicates that nothing was awaited ever
                // in the future.
                *this.state = GPUCommandBlockState::EarlyTerminated;
                return Some((output, retain));
            }
        }
        None
    }
}
