use std::alloc::{alloc, Layout};
use std::mem::MaybeUninit;
use std::num::NonZeroUsize;
use std::ops::{Index, IndexMut};

pub struct StackAllocator<T> {
    head: usize,
    arr: Box<[MaybeUninit<T>]>,
}

#[derive(Copy, Clone)]
pub struct StackAllocatorHandle {
    pub length: NonZeroUsize,
    start: usize,
}

impl<T: Default> StackAllocator<T> {
    pub fn new(len: usize) -> Self {
        StackAllocator {
            head: 0,
            arr: unsafe {
                let layout: Layout = Layout::new::<T>().repeat(len).unwrap().0;
                let ptr = alloc(layout) as *mut MaybeUninit<T>;
                let slice = std::slice::from_raw_parts_mut(ptr, len);
                Box::from_raw(slice)
            },
        }
    }

    #[inline]
    pub fn allocate(&mut self, length: usize) -> StackAllocatorHandle {
        assert!(
            length > 0,
            "Trying to allocate 0-lengthed memory in StackAllocator"
        );
        let start = self.head;
        let end = start + length;
        assert!(end <= self.arr.len(), "Allocator stack overflow");
        self.head += length;

        StackAllocatorHandle {
            length: unsafe { NonZeroUsize::new_unchecked(length) },
            start,
        }
    }

    #[inline]
    pub fn deallocate(&mut self, handle: StackAllocatorHandle) {
        assert_eq!(
            self.head,
            handle.start + handle.length.get(),
            "Trying to deallocate non-top items"
        );
        self.head -= handle.length.get();
    }

    #[inline]
    pub unsafe fn deallocate_size(&mut self, size: usize) {
        self.head -= size;
    }
}

impl<T> Index<StackAllocatorHandle> for StackAllocator<T> {
    type Output = [MaybeUninit<T>];

    fn index(&self, index: StackAllocatorHandle) -> &Self::Output {
        &self.arr[index.start..(index.start + index.length.get())]
    }
}

impl<T> IndexMut<StackAllocatorHandle> for StackAllocator<T> {
    fn index_mut(&mut self, index: StackAllocatorHandle) -> &mut Self::Output {
        &mut self.arr[index.start..(index.start + index.length.get())]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stack_allocator() {
        let mut allocator = StackAllocator::<u64>::new(10);
        let handle1 = allocator.allocate(3);
        let handle2 = allocator.allocate(6);
        let handle3 = allocator.allocate(1);

        allocator.deallocate(handle3);
        allocator.deallocate(handle2);
        allocator.deallocate(handle1);
    }
    #[test]
    #[should_panic(expected = "Trying to deallocate non-top items")]
    fn test_stack_allocator_wrong_order() {
        let mut allocator = StackAllocator::<u64>::new(10);
        let handle1 = allocator.allocate(3);
        let handle2 = allocator.allocate(6);
        let handle3 = allocator.allocate(1);

        allocator.deallocate(handle1);
    }
    #[test]
    #[should_panic(expected = "Allocator stack overflow")]
    fn test_stack_allocator_overflow() {
        let mut allocator = StackAllocator::<u64>::new(10);
        let handle1 = allocator.allocate(3);
        let handle2 = allocator.allocate(6);
        let handle3 = allocator.allocate(1);
        let handle4 = allocator.allocate(1);
    }
}
