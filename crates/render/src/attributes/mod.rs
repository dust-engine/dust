mod integer;
pub use integer::*;

use std::sync::Arc;

use dustash::resources::alloc::Allocator;

pub trait AttributeWriter<T> {
    fn new(allocator: &Arc<Allocator>, size: usize) -> Self;
    fn from_iter<I, II>(allocator: &Arc<Allocator>, into_iter: II) -> Self
        where I: ExactSizeIterator<Item = T>, II: IntoIterator<IntoIter = I>, Self: Sized {
            let iter = into_iter.into_iter();
            let mut this = Self::new(allocator, iter.len());
            for item in iter {
                this.write_item(item);
            }
            this
        }
    fn write_item(&mut self, item: T);

    /// Generally, this will be a GPU resource wrapped in Arc.
    /// The resource is wrapped in Arc because it might being asynchronously copied on the device side.
    type Resource;
    fn into_resource(self) -> Self::Resource;
}

