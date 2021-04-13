use super::stack::{StackAllocator, StackAllocatorHandle};
use crate::dir::Quadrant;
use std::mem::MaybeUninit;
use std::ops::Index;

#[derive(Copy, Clone)]
pub struct Surface {
    handle: StackAllocatorHandle,
    width: usize,
    offset_x: usize,
    offset_y: usize,
    pub(crate) size: usize,
}

impl Surface {
    pub fn new(handle: StackAllocatorHandle, size: usize) -> Surface {
        debug_assert_eq!(handle.length.get(), size * size);
        Surface {
            handle,
            width: size,
            offset_x: 0,
            offset_y: 0,
            size,
        }
    }

    #[inline]
    pub fn slice(&self, quadrant: Quadrant) -> Surface {
        let mut offset_x = self.offset_x;
        let mut offset_y = self.offset_y;
        let new_size = self.size / 2;
        assert!(new_size > 0, "Can't subdivide further!");
        let quadrant: u8 = quadrant as u8;
        if quadrant & 0b10 != 0 {
            // right
            offset_x += new_size;
        }
        if quadrant & 0b01 != 0 {
            // top
            offset_y += new_size;
        }
        Surface {
            handle: self.handle,
            width: self.width,
            offset_x,
            offset_y,
            size: new_size,
        }
    }

    #[inline]
    pub fn fill<T: Copy>(&self, allocator: &mut StackAllocator<T>, value: T) {
        let slice = &mut allocator[self.handle];
        let mut index = self.offset_y * self.width + self.offset_x;
        for _y in 0..self.size {
            for x in 0..self.size {
                slice[index + x].write(value);
            }
            index += self.width;
        }
    }

    #[inline]
    pub fn get<'a, T>(
        &self,
        allocator: &'a StackAllocator<T>,
        x: usize,
        y: usize,
    ) -> &'a MaybeUninit<T> {
        &allocator[self.handle][(y + self.offset_y) * self.width + self.offset_x + x]
    }

    #[inline]
    pub fn get_mut<'a, T>(
        &self,
        allocator: &'a mut StackAllocator<T>,
        x: usize,
        y: usize,
    ) -> &'a mut MaybeUninit<T> {
        &mut allocator[self.handle][(y + self.offset_y) * self.width + self.offset_x + x]
    }

    #[inline]
    pub fn get_first_row(&self) -> SurfaceRow {
        let index = self.offset_y * self.width + self.offset_x;
        SurfaceRow {
            index,
            width: self.width,
        }
    }

    #[inline]
    pub fn get_in_row<'a, T>(
        &self,
        allocator: &'a StackAllocator<T>,
        row: SurfaceRow,
        n: usize,
    ) -> &'a MaybeUninit<T> {
        &allocator[self.handle][row.index + n]
    }
}

#[derive(Copy, Clone)]
pub struct SurfaceRow {
    index: usize,
    width: usize,
}

impl SurfaceRow {
    pub fn next(&self) -> SurfaceRow {
        SurfaceRow {
            index: self.index + self.width,
            width: self.width,
        }
    }
}
