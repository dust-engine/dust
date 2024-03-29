use std::mem::size_of;

#[derive(Clone)]
pub struct BitMask<const SIZE: usize>
where
    [(); SIZE / size_of::<usize>() / 8]: Sized,
{
    pub(crate) data: [usize; SIZE / size_of::<usize>() / 8],
}

impl<const SIZE: usize> Default for BitMask<SIZE>
where
    [(); SIZE / size_of::<usize>() / 8]: Sized,
{
    fn default() -> Self {
        Self {
            data: [0; SIZE / size_of::<usize>() / 8],
        }
    }
}

impl<const SIZE: usize> std::fmt::Debug for BitMask<SIZE>
where
    [(); SIZE / size_of::<usize>() / 8]: Sized,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for item in self.data.iter() {
            f.write_fmt(format_args!("{:#064b}\n", item))?;
        }
        Ok(())
    }
}

impl<const SIZE: usize> BitMask<SIZE>
where
    [(); SIZE / size_of::<usize>() / 8]: Sized,
{
    pub fn new() -> Self {
        Self {
            data: [0; SIZE / size_of::<usize>() / 8],
        }
    }
    #[inline]
    pub fn get(&self, index: usize) -> bool {
        let i = index / size_of::<usize>() / 8;
        let j = index - i * size_of::<usize>() * 8;
        let val = self.data[i];

        let bit = (val >> j) & 1;
        bit != 0
    }
    #[inline]
    pub fn set(&mut self, index: usize, val: bool) {
        let i = index / size_of::<usize>() / 8;
        let j = index - i * size_of::<usize>() * 8;
        let entry = &mut self.data[i];
        if val {
            *entry |= 1 << j;
        } else {
            *entry &= !(1 << j);
        }
    }

    pub fn iter_set_bits(&self) -> SetBitIterator<SIZE> {
        SetBitIterator {
            bitmask: self,
            i: 0,
            state: self.data[0],
        }
    }

    pub fn is_zeroed(&self) -> bool {
        self.data.iter().all(|&a| a == 0)
    }

    pub fn count_ones(&self) -> usize {
        self.data.iter().map(|a| a.count_ones() as usize).sum()
    }
}

/// ```
/// #![feature(generic_const_exprs)]
/// let mut bitmask = dust_vdb::BitMask::<128>::new();
/// bitmask.set(12, true);
/// bitmask.set(101, true);
/// let mut iter = bitmask.iter_set_bits();
/// assert_eq!(iter.next(), Some(12));
/// assert_eq!(iter.next(), Some(101));
/// assert!(iter.next().is_none());
/// ```
pub struct SetBitIterator<'a, const SIZE: usize>
where
    [(); SIZE / size_of::<usize>() / 8]: Sized,
{
    bitmask: &'a BitMask<SIZE>,
    i: usize,
    state: usize,
}

impl<'a, const SIZE: usize> Iterator for SetBitIterator<'a, SIZE>
where
    [(); SIZE / size_of::<usize>() / 8]: Sized,
{
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        const NUM_BITS: usize = std::mem::size_of::<usize>() * 8;
        loop {
            if self.state == 0 {
                if self.i == self.bitmask.data.len() - 1 {
                    return None;
                }
                self.i += 1;
                self.state = self.bitmask.data[self.i];
                continue;
            }

            let t = self.state & (!self.state).wrapping_add(1);
            let r = self.state.trailing_zeros() as usize;
            let result = self.i * NUM_BITS + r;
            self.state ^= t;
            return Some(result);
        }
    }
}
