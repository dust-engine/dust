use crate::Corner;
use std::fmt::Write;
use std::num::NonZeroU64;

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct IndexPath(NonZeroU64);

impl IndexPath {
    const MAX_SIZE: u8 = 21;

    pub fn new() -> Self {
        unsafe { Self::from(NonZeroU64::new_unchecked(1)) }
    }

    pub fn is_empty(&self) -> bool {
        Into::<u64>::into(*self) == 1
    }
    pub fn is_full(&self) -> bool {
        // Check highest bit
        (Into::<u64>::into(*self) >> 63) == 1
    }
    pub fn peek(&self) -> Corner {
        assert!(!self.is_empty());
        let val: u8 = (self.0.get() & 0b111) as u8;
        val.into()
    }
    pub fn pop(&self) -> Self {
        assert!(!self.is_empty());
        let num = self.0.get() >> 3;
        unsafe { IndexPath(NonZeroU64::new_unchecked(num)) }
    }
    pub fn push(&self, octant: Corner) -> Self {
        assert!(!self.is_full(), "The index path is full");
        let num = (self.0.get() << 3) | (octant as u64);
        unsafe { IndexPath(NonZeroU64::new_unchecked(num)) }
    }
    pub fn count(&self) -> u8 {
        Self::MAX_SIZE - (Into::<u64>::into(*self).leading_zeros() / 3) as u8
    }
    pub fn replace(&self, octant: Corner) -> Self {
        unsafe {
            let num: u64 = self.0.get() & !0b111;
            Self::from(NonZeroU64::new_unchecked(num | (octant as u64)))
        }
    }
    pub fn len(&self) -> u8 {
        let num_empty_slots = self.0.get().leading_zeros() as u8 / 3;
        Self::MAX_SIZE - num_empty_slots
    }
}

impl From<NonZeroU64> for IndexPath {
    fn from(val: NonZeroU64) -> Self {
        Self(val)
    }
}
impl From<IndexPath> for NonZeroU64 {
    fn from(index_path: IndexPath) -> NonZeroU64 {
        index_path.0
    }
}
impl From<IndexPath> for u64 {
    fn from(index_path: IndexPath) -> u64 {
        index_path.0.get()
    }
}

impl Iterator for IndexPath {
    type Item = Corner;

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_empty() {
            None
        } else {
            let dir = self.peek();
            self.0 = self.pop().0;
            Some(dir)
        }
    }
}

impl std::fmt::Debug for IndexPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        let mut current = self.clone();
        f.write_str("(Root)")?;
        while !current.is_empty() {
            f.write_char('/')?;
            f.write_char((current.peek() as u8 + '0' as u8).into())?;
            current = current.pop();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    #[test]
    fn test_index_path() {
        assert_eq!(size_of::<IndexPath>(), size_of::<u64>());
        assert_eq!(size_of::<Option<IndexPath>>(), size_of::<u64>());

        let mut path = IndexPath::new();
        for i in 0..IndexPath::MAX_SIZE {
            assert_eq!(path.len(), i);
            path = path.push(Corner::FrontLeftBottom);
        }
        assert_eq!(path.len(), IndexPath::MAX_SIZE);
    }
}
