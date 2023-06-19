/// Given iterator of the form `Iterator<Item=uxx>`
/// emitting ordered items,
/// `MergeRangeIterator` emits list of (start, size)
/// for each consecutive range of numbers, where `start`
/// is the starting indice of the range, and `size` is the
/// number of consecutive items.
pub struct MergeRangeIterator<ITER: Iterator<Item = usize>> {
    inner: ITER,
    last: Option<usize>,
}

impl<ITER: Iterator<Item = usize>> Iterator for MergeRangeIterator<ITER> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        // (start, size)
        let mut state: Option<(usize, usize)> = self.last.take().map(|last| (last, 1));
        loop {
            let Some(next) = self.inner.next() else {
                return state;
            };

            if let Some((start, size)) = state.as_mut() {
                if next == *start + *size {
                    *size += 1;
                    continue;
                } else {
                    self.last = Some(next);
                    return Some((*start, *size));
                }
            } else {
                state = Some((next, 1));
            }
        }
    }
}

pub trait MergeRangeIteratorExt {
    fn merge_ranges(self) -> MergeRangeIterator<Self>
    where
        Self: Iterator<Item = usize> + Sized,
    {
        MergeRangeIterator {
            inner: self,
            last: None,
        }
    }
}
impl<T: Iterator<Item = usize> + Sized> MergeRangeIteratorExt for T {}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_empty() {
        let mut iter = [].into_iter().merge_ranges();
        assert!(iter.next().is_none());
    }
    #[test]
    fn test_one() {
        let mut iter = [10].into_iter().merge_ranges();
        assert_eq!(iter.next().unwrap(), (10, 1));
        assert!(iter.next().is_none());
    }

    #[test]
    fn test() {
        let mut iter = [11, 12, 15, 16, 17].into_iter().merge_ranges();
        assert_eq!(iter.next().unwrap(), (11, 2));
        assert_eq!(iter.next().unwrap(), (15, 3));
        assert!(iter.next().is_none());
    }
}
