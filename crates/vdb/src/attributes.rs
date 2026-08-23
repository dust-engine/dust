use glam::UVec3;

pub trait IsDefault {
    fn is_default(&self) -> bool;
}
impl<T> IsDefault for T
where
    T: Default + Eq,
{
    fn is_default(&self) -> bool {
        self == &Self::default()
    }
}

pub trait Attributes {
    /// The type of the attribute pointer.
    /// The attribute pointers are stored on the vdb leaf nodes, one per node.
    /// This is typically u32.
    type Ptr;
    /// The type of the attribute values. For a MagicaVoxel grid, this would be a u8 palette index.
    type Value: Default + IsDefault;
    fn get_attribute(&self, leaf: u32, ptr: &Self::Ptr, offset: u32) -> Self::Value;
    fn set_attribute(&mut self, leaf: u32, ptr: &Self::Ptr, offset: u32, value: Self::Value);
    fn free_attributes(&mut self, leaf: u32, ptr: &Self::Ptr, num_attributes: u32);

    /// Allocate a new attribute range using the new mask. Then, copy the attributes from the attribute range
    /// pointed to by `ptr` to the newly allocated attribute range. Returns the pointer to the new attribute range.
    ///
    /// Only attribute values that are set in both the original mask and the new mask will be copied.
    ///
    /// The original attribute range will not be freed. It is the responsibility of the caller to free the original attribute range.
    ///
    /// Note that the original mask may be zeroed. In this case, `ptr` is meaningless, and the function will allocate
    /// a new attribute range without performing any copy.
    ///
    /// `original_leaf` and `new_leaf` are the pool indices of the leaf whose
    /// range is being copied from and of the leaf that will own the new range.
    /// They are equal when a leaf re-homes its own range (inflating on
    /// becoming the hot leaf, fitting on leaving); they differ when an edit
    /// forked a snapshot-shared leaf (copy-on-write). Implementations that
    /// key their ranges by leaf index instead of through `ptr` — e.g.
    /// hierarchy-erased side tables — need them; others may ignore them.
    ///
    /// Occupancy masks are the leaf's raw occupancy words — word-aligned by
    /// construction, whatever the hierarchy's leaf size: bit `i` of a mask
    /// is bit `i % 64` of word
    /// `i / 64`. Walk them with [`iter_mask_union`]; count them with
    /// [`mask_count_ones`]. The all-ones mask of a fully inflated range is
    /// supplied by the caller (the accessor takes it from
    /// [`IsLeaf::MAX_OCCUPANCY`](crate::IsLeaf)), so implementations —
    /// including hierarchy-erased ones that cannot name a leaf size at
    /// compile time — never construct one.
    fn copy_attribute(
        &mut self,
        original_leaf: u32,
        new_leaf: u32,
        ptr: &Self::Ptr,
        original_mask: &[usize],
        new_mask: &[usize],
        coords: &UVec3,
    ) -> Self::Ptr; // need a value to represent: what are the ones to delete, and what are the ones to add?
}

/// Number of set bits in an occupancy mask.
pub fn mask_count_ones(words: &[usize]) -> u32 {
    words.iter().map(|w| w.count_ones()).sum()
}

/// Walks the union of two occupancy masks in ascending bit order, yielding
/// `(bit, in_original, in_new)` for every position set in either mask.
///
/// This is the iteration [`Attributes::copy_attribute`] implementations need
/// to rank-walk two fitted ranges: advance the source cursor on
/// `in_original`, the destination cursor on `in_new`, and copy when both are
/// set.
///
/// The walk is word-at-a-time: each word pair costs two loads and one `or`,
/// each set bit of the union one `trailing_zeros`, one clear-lowest-bit, and
/// two `and`s — zero bits are skipped a word at a time, and nothing is
/// allocated. A mask reads as zero past its end, so the masks may differ in
/// length.
pub fn iter_mask_union<'a>(original: &'a [usize], new: &'a [usize]) -> MaskUnionIter<'a> {
    let mut iter = MaskUnionIter {
        original,
        new,
        union_word: 0,
        original_word: 0,
        new_word: 0,
        word_index: 0,
        num_words: original.len().max(new.len()),
    };
    if iter.num_words > 0 {
        iter.original_word = original.first().copied().unwrap_or(0);
        iter.new_word = new.first().copied().unwrap_or(0);
        iter.union_word = iter.original_word | iter.new_word;
    }
    iter
}

/// See [`iter_mask_union`].
pub struct MaskUnionIter<'a> {
    original: &'a [usize],
    new: &'a [usize],
    /// Union of the current word pair, already-yielded bits cleared.
    union_word: usize,
    original_word: usize,
    new_word: usize,
    word_index: usize,
    num_words: usize,
}

impl Iterator for MaskUnionIter<'_> {
    type Item = (usize, bool, bool);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.union_word == 0 {
            let next_word = self.word_index + 1;
            if next_word >= self.num_words {
                return None;
            }
            self.word_index = next_word;
            self.original_word = self.original.get(next_word).copied().unwrap_or(0);
            self.new_word = self.new.get(next_word).copied().unwrap_or(0);
            self.union_word = self.original_word | self.new_word;
        }
        let bit_in_word = self.union_word.trailing_zeros() as usize;
        let mask = 1usize << bit_in_word;
        self.union_word &= self.union_word - 1;
        Some((
            self.word_index * usize::BITS as usize + bit_in_word,
            self.original_word & mask != 0,
            self.new_word & mask != 0,
        ))
    }
}

/// Virtual buffer designed specifically for allocating attributes.
pub struct AttributeAllocator {
    freelists: Box<[Vec<u32>]>,
    alignment: u32,
    max_allocation: u32,
    head: u32,
    wasted_bytes: u32,
}

impl AttributeAllocator {
    fn freelist_for_size(&mut self, size: u32) -> &mut Vec<u32> {
        let freelist_index = (size - 1) / self.alignment;
        &mut self.freelists[freelist_index as usize]
    }
    pub fn new_with_capacity(alignment: u32, max_allocation: u32) -> Self {
        let num_freelists = max_allocation.div_ceil(alignment);
        Self {
            alignment,
            max_allocation,
            freelists: vec![Vec::new(); num_freelists as usize].into_boxed_slice(),
            head: 0,
            wasted_bytes: 0,
        }
    }
    pub fn allocate(&mut self, size: u32) -> u32 {
        assert!(size <= self.max_allocation);
        assert!(size > 0);
        let increment = size.next_multiple_of(self.alignment);
        self.wasted_bytes += increment - size;
        if let Some(indice) = self.freelist_for_size(size).pop() {
            return indice;
        }
        let old_head = self.head;
        self.head += increment;
        return old_head;
    }
    pub fn realloc(&mut self, ptr: u32, old_size: u32, new_size: u32) -> u32 {
        let old_increment = old_size.next_multiple_of(self.alignment);
        let new_increment = new_size.next_multiple_of(self.alignment);
        if old_increment == new_increment {
            return ptr;
        }
        self.free(ptr, old_size);
        self.allocate(new_size)
    }
    pub fn free(&mut self, ptr: u32, size: u32) {
        assert!(size <= self.max_allocation);
        self.freelist_for_size(size).push(ptr);
        self.wasted_bytes -= size.next_multiple_of(self.alignment) - size;
    }
}

#[cfg(test)]
mod tests {
    use super::{iter_mask_union, mask_count_ones};

    /// Reference implementation: test every bit of both masks directly.
    fn naive(original: &[usize], new: &[usize]) -> Vec<(usize, bool, bool)> {
        let bits = original.len().max(new.len()) * usize::BITS as usize;
        (0..bits)
            .filter_map(|bit| {
                let word = bit / usize::BITS as usize;
                let mask = 1usize << (bit % usize::BITS as usize);
                let in_original = original.get(word).is_some_and(|w| w & mask != 0);
                let in_new = new.get(word).is_some_and(|w| w & mask != 0);
                (in_original || in_new).then_some((bit, in_original, in_new))
            })
            .collect()
    }

    #[test]
    fn test_mask_union() {
        // Irregular patterns, including an all-zero word pair to skip whole.
        let a = [0b1010_1001usize, 0, (1 << 63) | (1 << 2)];
        let b = [0b0110_0000usize, 0, 1 << 2];
        assert_eq!(iter_mask_union(&a, &b).collect::<Vec<_>>(), naive(&a, &b));

        // Different lengths: positions past the shorter mask read as unset.
        assert_eq!(
            iter_mask_union(&a[..1], &b).collect::<Vec<_>>(),
            naive(&a[..1], &b)
        );
        assert_eq!(iter_mask_union(&a, &[]).collect::<Vec<_>>(), naive(&a, &[]));

        // Empty and all-zero masks yield nothing.
        assert_eq!(iter_mask_union(&[], &[]).count(), 0);
        assert_eq!(iter_mask_union(&[0, 0], &[0, 0]).count(), 0);

        assert_eq!(mask_count_ones(&a), 6);
    }
}
