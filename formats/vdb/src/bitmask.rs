use std::mem::size_of;

pub struct BitMask<const SIZE: usize>
where
    [(); SIZE / size_of::<usize>()]: Sized,
{
    data: [usize; SIZE / size_of::<usize>()],
}

impl<const SIZE: usize> Default for BitMask<SIZE>
where
    [(); SIZE / size_of::<usize>()]: Sized,
{
    fn default() -> Self {
        Self {
            data: [0; SIZE / size_of::<usize>()],
        }
    }
}

impl<const SIZE: usize> BitMask<SIZE>
where
    [(); SIZE / size_of::<usize>()]: Sized,
{
    pub fn new() -> Self {
        Self {
            data: [0; SIZE / size_of::<usize>()],
        }
    }
    #[inline]
    pub fn get(&self, index: usize) -> bool {
        let i = index / size_of::<usize>();
        let j = index - i * size_of::<usize>();
        let val = self.data[i];

        let bit = (val >> j) & 1;
        bit != 0
    }
    #[inline]
    pub fn set(&mut self, index: usize, val: bool) {
        let i = index / size_of::<usize>();
        let j = index - i * size_of::<usize>();
        let entry = &mut self.data[i];
        if val {
            *entry |= 1 << j;
        } else {
            *entry &= !(1 << j);
        }
    }
}
