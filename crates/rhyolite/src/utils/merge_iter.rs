use core::cmp::Ordering;
use core::fmt::{self, Debug};
use core::iter::FusedIterator;
use std::cmp::max;
use std::collections::btree_map::Iter;
use std::collections::BTreeMap;

/// Core of an iterator that merges the output of two strictly ascending iterators,
/// for instance a union or a symmetric difference.
pub struct MergeIterInner<I1: Iterator, I2: Iterator> {
    a: I1,
    b: I2,
    peeked: Option<Peeked<I1, I2>>,
}

/// Benchmarks faster than wrapping both iterators in a Peekable,
/// probably because we can afford to impose a FusedIterator bound.
#[derive(Clone, Debug)]
enum Peeked<I1: Iterator, I2: Iterator> {
    A(I1::Item),
    B(I2::Item),
}

impl<I1: Iterator, I2: Iterator> Clone for MergeIterInner<I1, I2>
where
    I1: Clone,
    I1::Item: Clone,
    I2: Clone,
    I2::Item: Clone,
{
    fn clone(&self) -> Self {
        Self {
            a: self.a.clone(),
            b: self.b.clone(),
            peeked: self.peeked.clone(),
        }
    }
}

impl<I1: Iterator, I2: Iterator> Debug for MergeIterInner<I1, I2>
where
    I1: Debug,
    I1::Item: Debug,
    I2: Debug,
    I2::Item: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("MergeIterInner")
            .field(&self.a)
            .field(&self.b)
            .field(&self.peeked)
            .finish()
    }
}

impl<I1: Iterator, I2: Iterator> MergeIterInner<I1, I2> {
    /// Creates a new core for an iterator merging a pair of sources.
    pub fn new(a: I1, b: I2) -> Self {
        MergeIterInner { a, b, peeked: None }
    }

    /// Returns the next pair of items stemming from the pair of sources
    /// being merged. If both returned options contain a value, that value
    /// is equal and occurs in both sources. If one of the returned options
    /// contains a value, that value doesn't occur in the other source (or
    /// the sources are not strictly ascending). If neither returned option
    /// contains a value, iteration has finished and subsequent calls will
    /// return the same empty pair.
    pub fn nexts<Cmp: Fn(&I1::Item, &I2::Item) -> Ordering>(
        &mut self,
        cmp: Cmp,
    ) -> (Option<I1::Item>, Option<I2::Item>)
    where
        I1: FusedIterator,
        I2: FusedIterator,
    {
        let mut a_next;
        let mut b_next;
        match self.peeked.take() {
            Some(Peeked::A(next)) => {
                a_next = Some(next);
                b_next = self.b.next();
            }
            Some(Peeked::B(next)) => {
                b_next = Some(next);
                a_next = self.a.next();
            }
            None => {
                a_next = self.a.next();
                b_next = self.b.next();
            }
        }
        if let (Some(ref a1), Some(ref b1)) = (&a_next, &b_next) {
            match cmp(a1, b1) {
                Ordering::Less => self.peeked = b_next.take().map(Peeked::B),
                Ordering::Greater => self.peeked = a_next.take().map(Peeked::A),
                Ordering::Equal => (),
            }
        }
        (a_next, b_next)
    }

    /// Returns a pair of upper bounds for the `size_hint` of the final iterator.
    pub fn lens(&self) -> (usize, usize)
    where
        I1: ExactSizeIterator,
        I2: ExactSizeIterator,
    {
        match self.peeked {
            Some(Peeked::A(_)) => (1 + self.a.len(), self.b.len()),
            Some(Peeked::B(_)) => (self.a.len(), 1 + self.b.len()),
            _ => (self.a.len(), self.b.len()),
        }
    }
}

pub struct Union<'a, K: 'a, V1: 'a, V2: 'a> {
    inner: MergeIterInner<Iter<'a, K, V1>, Iter<'a, K, V2>>,
}

impl<'a, K: Ord, V1, V2> Iterator for Union<'a, K, V1, V2> {
    type Item = (&'a K, Option<&'a V1>, Option<&'a V2>);

    fn next(&mut self) -> Option<Self::Item> {
        let (a_next, b_next) = self.inner.nexts(|a, b| a.0.cmp(b.0));
        let key = a_next.map(|a| a.0).or(b_next.map(|b| b.0));
        if let Some(key) = key {
            let a_next = a_next.map(|a| a.1);
            let b_next = b_next.map(|b| b.1);
            Some((key, a_next, b_next))
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (a_len, b_len) = self.inner.lens();
        // No checked_add - see SymmetricDifference::size_hint.
        (max(a_len, b_len), Some(a_len + b_len))
    }

    fn min(mut self) -> Option<Self::Item> {
        self.next()
    }
}

pub fn btree_map_union<'a, K, V1, V2>(
    a: &'a BTreeMap<K, V1>,
    b: &'a BTreeMap<K, V2>,
) -> Union<'a, K, V1, V2> {
    Union {
        inner: MergeIterInner::new(a.iter(), b.iter()),
    }
}
