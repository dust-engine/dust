use std::{marker::{PhantomData}, alloc::Layout, mem::ManuallyDrop, ops::{Index, IndexMut}, ptr::{NonNull, Pointee}};

pub struct UnsizedVec<T: ?Sized> {
    metadata: Vec<<T as Pointee>::Metadata>,
    inner: Vec<u8>,
    layout: Layout,
    _marker: PhantomData<T>
}

impl<T: ?Sized> UnsizedVec<T> {
    pub fn push(&mut self, mut item: T) {
        let item_layout = std::alloc::Layout::for_value(&item);
        assert!(item_layout.pad_to_align().size() <= self.layout.pad_to_align().size());

        let ptr = unsafe {
            NonNull::new_unchecked(&mut item)
        };
        let (raw_ptr, metadata) = ptr.to_raw_parts();
        self.metadata.push(metadata);

        let offset = self.inner.len();
        self.inner.extend(std::iter::repeat(0).take(self.layout.pad_to_align().size()));
        unsafe {
            std::ptr::copy_nonoverlapping(raw_ptr.cast::<u8>().as_ptr(), self.inner.as_mut_ptr().add(offset), item_layout.size());
        }
        std::mem::forget_unsized(item);
    }
    pub fn len(&self) -> usize {
        self.metadata.len()
    }
    pub fn clear(&mut self) {
        for i in 0..self.metadata.len() {
            unsafe {
                std::ptr::drop_in_place(&mut self[i]);
            }
        }
        self.metadata.clear();
        self.inner.clear();
    }
}

impl<T: ?Sized> Drop for UnsizedVec<T> {
    fn drop(&mut self) {
        for i in 0..self.metadata.len() {
            unsafe {
                std::ptr::drop_in_place(&mut self[i]);
            }
        }
    }
}




impl<T: ?Sized> Index<usize> for UnsizedVec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        let metadata = self.metadata[index];
        let index = index * self.layout.pad_to_align().size();
        unsafe {
            let ptr = self.inner.as_ptr().add(index);
            let ptr: NonNull<()> = NonNull::new_unchecked(ptr as *mut u8).cast();
            let ptr: NonNull<T> = NonNull::from_raw_parts(ptr, metadata);
            ptr.as_ref()
        }
    }
}
impl<T: ?Sized> IndexMut<usize> for UnsizedVec<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let metadata = self.metadata[index];
        let index = index * self.layout.pad_to_align().size();
        unsafe {
            let ptr = self.inner.as_mut_ptr().add(index);
            let ptr: NonNull<()> = NonNull::new_unchecked(ptr).cast();
            let mut ptr: NonNull<T> = NonNull::from_raw_parts(ptr, metadata);
            ptr.as_mut()
        }
    }
}

