use std::marker::PhantomData;
use super::BLOCK_SIZE;

#[derive(Copy, Clone)]
pub struct Handle<T: Copy> {
    _marker: PhantomData<T>,
    pub(crate) index: u32,
}

impl<T: Copy> Handle<T> {
    pub const fn none() -> Self {
        Handle {
            _marker: PhantomData,
            index: std::u32::MAX,
        }
    }
    #[inline]
    pub fn is_none(&self) -> bool {
        self.index == std::u32::MAX
    }
    pub(crate) fn new(block_num: u32, item_num: u32) -> Self {
        Handle {
            _marker: PhantomData,
            index: (block_num << BLOCK_SIZE) | item_num,
        }
    }
    pub(crate) fn offset(&self, n: u32) -> Self {
        let (block_num, item_num): (u32, u32) = self.into();
        Handle::new(block_num, item_num + n)
    }
}

impl<T: Copy> From<Handle<T>> for (u32, u32) {
    fn from(handle: Handle<T>) -> Self {
        let item_num = handle.index & ((1 << BLOCK_SIZE) - 1);
        let block_num = handle.index >> BLOCK_SIZE;
        (block_num, item_num)
    }
}

impl<T: Copy> From<&Handle<T>> for (u32, u32) {
    fn from(handle: &Handle<T>) -> Self {
        (*handle).into()
    }
}

impl<T: Copy> From<Handle<T>> for (usize, usize) {
    fn from(handle: Handle<T>) -> Self {
        let (block_num, item_num): (u32, u32) = handle.into();
        (block_num as usize, item_num as usize)
    }
}

impl<T: Copy> std::fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        let (block_num, item_num): (u32, u32) = self.into();
        f.write_fmt(format_args!("Handle({:?}, {:?})", block_num, item_num))
    }
}

impl<T: Copy> std::cmp::PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl<T: Copy> Eq for Handle<T> {}
