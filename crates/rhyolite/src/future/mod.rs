use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::pin::Pin;
use std::task::Poll;

mod block;
mod exec;
mod ext;
mod state;
pub use block::*;
pub use exec::*;
pub use ext::*;
use pin_project::pin_project;
pub use state::*;

// TODO: Use the dispose crate.
pub trait Disposable {
    fn retire(&mut self) {}
    fn dispose(self);
}
pub struct Dispose<T>(PhantomData<T>);
impl<T> Dispose<T> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}
impl<T> Disposable for Dispose<T> {
    fn dispose(self) {
        std::mem::forget(self)
    }
}
impl<T> Drop for Dispose<T> {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            panic!("Dispose<{}> must be disposed!", std::any::type_name::<T>());
        }
    }
}

pub struct DisposeContainer<T> {
    inner: ManuallyDrop<T>,
    marker: Dispose<T>,
}
impl<T> DisposeContainer<T> {
    pub fn new(item: T) -> Self {
        Self {
            inner: ManuallyDrop::new(item),
            marker: Dispose::new(),
        }
    }
}
impl<T> Disposable for DisposeContainer<T> {
    fn dispose(mut self) {
        unsafe {
            ManuallyDrop::drop(&mut self.inner);
        }
        self.marker.dispose();
    }
}

impl Disposable for () {
    fn dispose(self) {}
}
impl Disposable for Box<dyn Disposable> {
    fn retire(&mut self) {
        (**self).retire()
    }
    fn dispose(self) {
        (*self).dispose()
    }
}

impl Disposable for Box<dyn Disposable + Send> {
    fn retire(&mut self) {
        (**self).retire()
    }
    fn dispose(self) {
        (*self).dispose()
    }
}
impl<T: Disposable> Disposable for Vec<T> {
    fn retire(&mut self) {
        for i in self.iter_mut() {
            i.retire();
        }
    }
    fn dispose(self) {
        for i in self.into_iter() {
            i.dispose();
        }
    }
}

macro_rules! impl_tuple {
    ($($idx:tt $t:tt),+) => {
        impl<$($t,)+> Disposable for ($($t,)+)
        where
            $($t: Disposable,)+
        {
            fn retire(&mut self) {
                $(
                    $t :: retire(&mut self.$idx);
                )+
            }
            fn dispose(self) {
                $(
                    $t :: dispose(self.$idx);
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
impl_tuple!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G, 7 H, 8 I);
impl_tuple!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G, 7 H, 8 I, 9 J);
impl_tuple!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G, 7 H, 8 I, 9 J, 10 K);
impl_tuple!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G, 7 H, 8 I, 9 J, 10 K, 11 L);

impl<T> Disposable for Option<T>
where
    T: Disposable,
{
    fn retire(&mut self) {
        if let Some(this) = self {
            this.retire()
        }
    }
    fn dispose(self) {
        if let Some(this) = self {
            this.dispose()
        }
    }
}

pub trait GPUCommandFuture {
    type Output;

    /// Objects with lifetimes that need to be extended until the future was executed on the GPU.
    type RetainedState: Disposable + Send;

    /// Optional object to be passed in at record time that collects reused states.
    type RecycledState: Default + Send + Sync;

    /// Attempt to record as many commands as possible into the provided
    /// command_buffer until a pipeline barrier is needed.
    ///
    /// Commands recorded inbetween two pipeline barriers are considered
    /// to be in the same "stage". These commands do not have any dependencies
    /// between each other, and they should run independently of each other.
    ///
    /// # Return value
    ///
    /// This function returns:
    ///
    /// - [`Poll::Pending`] if it's possible to record more commands
    /// - [`Poll::Ready(val)`] with the return value `val` of this future,
    ///   if no more commands can be recorded.
    ///
    /// Once a future has finished, clients should not call `record` on it again.
    ///
    /// # Runtime characteristics
    ///
    /// Futures alone are *inert*; they must be `record`ed into a command
    /// buffer and submitted to a queue in order to do work on the GPU.
    fn record(
        self: Pin<&mut Self>,
        ctx: &mut CommandBufferRecordContext,
        recycled_state: &mut Self::RecycledState,
    ) -> Poll<(Self::Output, Self::RetainedState)>;

    /// Returns the context for the operations recorded into the command buffer
    /// next time `record` was called.
    fn context(self: Pin<&mut Self>, ctx: &mut StageContext);

    /// Initialize the pinned future.
    /// This method is mostly a hook for `GPUCommandFutureBlock` to move forward to its first
    /// yield point. `GPUCommandFutureBlock` would then yields the function pointer to its
    /// first future to be awaited, allowing us to call the `context` method to retrieve the
    /// context.
    ///
    /// Returns a boolean indicating if this future should be run. If the implementation returns
    /// false, the entire future will be skipped, and no further calls to `record` or `context`
    /// will be made.
    ///
    /// For executors, this method should be called once, and as soon as the future was pinnned.
    /// For implementations of `GPUCommandFuture`, this method can be ignored in most cases.
    /// For combinators, this method should be called recursively for all inner futures.
    fn init(
        self: Pin<&mut Self>,
        _ctx: &mut CommandBufferRecordContext,
        _recycled_state: &mut Self::RecycledState,
    ) -> Option<(Self::Output, Self::RetainedState)> {
        None
    }
}

#[pin_project]
pub struct UnitCommandFuture<T> {
    obj: Option<T>,
}
impl<T> UnitCommandFuture<T> {
    pub fn new(obj: T) -> Self {
        UnitCommandFuture { obj: Some(obj) }
    }
}
impl<T> GPUCommandFuture for UnitCommandFuture<T> {
    type Output = T;
    type RecycledState = ();
    type RetainedState = ();
    fn record(
        self: Pin<&mut Self>,
        _ctx: &mut CommandBufferRecordContext,
        _recycled_state: &mut Self::RecycledState,
    ) -> Poll<(Self::Output, Self::RetainedState)> {
        let this = self.project();
        Poll::Ready((this.obj.take().unwrap(), ()))
    }
    fn context(self: Pin<&mut Self>, _ctx: &mut StageContext) {}
}
