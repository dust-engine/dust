use std::{mem::size_of, ops::BitOr, process::Output};

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
        if i >= self.data.len() {
            return false;
        }
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

    pub fn iter_set_bits(&self) -> SetBitIterator<std::iter::Cloned<std::slice::Iter<usize>>> {
        let mut iter = self.data.iter().cloned();
        SetBitIterator {
            state: iter.next().unwrap_or(0),
            inner: iter,
            i: 0,
        }
    }

    pub fn is_zeroed(&self) -> bool {
        self.data.iter().all(|&a| a == 0)
    }

    pub fn is_maxed(&self) -> bool {
        self.data.iter().all(|&a| a == usize::MAX)
    }
    pub fn count_ones(&self) -> usize {
        self.data.iter().map(|a| a.count_ones() as usize).sum()
    }
    pub fn as_slice(&self) -> &[usize] {
        &self.data
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
pub struct SetBitIterator<T: Iterator<Item = usize>> {
    inner: T,
    i: usize,
    state: usize,
}

impl<T: Iterator<Item = usize>> Iterator for SetBitIterator<T> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        const NUM_BITS: usize = std::mem::size_of::<usize>() * 8;
        loop {
            if self.state == 0 {
                self.i += 1;
                self.state = self.inner.next()?;
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

pub struct OrredBitMask<'a, const SIZE: usize>
where
    [(); SIZE / size_of::<usize>() / 8]: Sized,
{
    left: &'a [usize; SIZE / size_of::<usize>() / 8],
    right: &'a [usize; SIZE / size_of::<usize>() / 8],
}
impl<'a, const SIZE: usize> BitOr for &'a BitMask<SIZE>
where
    [(); SIZE / size_of::<usize>() / 8]: Sized,
{
    type Output = OrredBitMask<'a, SIZE>;
    fn bitor(self, rhs: Self) -> Self::Output {
        OrredBitMask {
            left: &self.data,
            right: &rhs.data,
        }
    }
}
impl<'a, const SIZE: usize> OrredBitMask<'a, SIZE>
where
    [(); SIZE / size_of::<usize>() / 8]: Sized,
{
    pub fn iter_set_bits(&self) -> SetBitIterator<impl Iterator<Item = usize> + '_> {
        let mut iter = self.left.iter().zip(self.right.iter()).map(|(a, b)| a | b);
        SetBitIterator {
            state: iter.next().unwrap_or(0),
            inner: iter,
            i: 0,
        }
    }
}

pub trait IsBitMask {
    const MAXED: Self;
    fn is_maxed(&self) -> bool;
}

impl<const SIZE: usize> IsBitMask for BitMask<SIZE>
where
    [(); SIZE / size_of::<usize>() / 8]: Sized,
{
    const MAXED: Self = BitMask {
        data: [usize::MAX; SIZE / size_of::<usize>() / 8],
    };
    fn is_maxed(&self) -> bool {
        self.is_maxed()
    }
}
