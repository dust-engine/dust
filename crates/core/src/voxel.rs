#[derive(Copy, Clone, PartialEq, Eq, Debug, Ord, PartialOrd)]
pub struct Voxel(u16);
impl Voxel {
    pub const AIR: Voxel = Voxel(0);
    pub fn with_id(id: u16) -> Voxel {
        Voxel(id)
    }
}
impl Default for Voxel {
    fn default() -> Self {
        Voxel::AIR
    }
}

impl svo::Voxel for Voxel {
    fn avg(arr: &[Self; 8]) -> Self {
        // find most frequent element
        let mut arr = arr.clone();
        arr.sort();
        if arr[7] == Self::default() {
            return Self::default();
        }

        let mut count: u8 = 1;
        let mut max_count: u8 = 0;
        let mut max_element: Self = Voxel(0);
        let mut last_element: Self;

        let mut iter = arr.iter();
        loop {
            if let Some(&val) = iter.next() {
                if val == Self::default() {
                    continue;
                } else {
                    last_element = val;
                    break;
                }
            } else {
                // never found a non-zero element until the very end.
                return Self::default();
            }
        }

        for &i in iter {
            if i != last_element {
                if count > max_count {
                    max_count = count;
                    max_element = last_element;
                }
                count = 0;
                last_element = i;
            }
            count += 1;
        }
        if count > max_count {
            max_element = last_element;
        }
        max_element
    }
}

#[cfg(test)]
mod tests {
    use super::Voxel;
    use svo::Voxel as VoxelTrait;

    #[test]
    fn test() {
        assert_eq!(
            Voxel::avg(&[
                Voxel::with_id(0),
                Voxel::with_id(0),
                Voxel::with_id(0),
                Voxel::with_id(3),
                Voxel::with_id(0),
                Voxel::with_id(0),
                Voxel::with_id(0),
                Voxel::with_id(0),
            ]),
            Voxel::with_id(3)
        );
        assert_eq!(
            Voxel::avg(&[
                Voxel::with_id(0),
                Voxel::with_id(0),
                Voxel::with_id(0),
                Voxel::with_id(3),
                Voxel::with_id(3),
                Voxel::with_id(3),
                Voxel::with_id(4),
                Voxel::with_id(0),
            ]),
            Voxel::with_id(3)
        );
        assert_eq!(
            Voxel::avg(&[
                Voxel::with_id(0),
                Voxel::with_id(0),
                Voxel::with_id(0),
                Voxel::with_id(0),
                Voxel::with_id(0),
                Voxel::with_id(0),
                Voxel::with_id(0),
                Voxel::with_id(0),
            ]),
            Voxel::with_id(0)
        );
        assert_eq!(
            Voxel::avg(&[
                Voxel::with_id(1),
                Voxel::with_id(2),
                Voxel::with_id(3),
                Voxel::with_id(4),
                Voxel::with_id(5),
                Voxel::with_id(6),
                Voxel::with_id(7),
                Voxel::with_id(8),
            ]),
            Voxel::with_id(1)
        );
        assert_eq!(
            Voxel::avg(&[
                Voxel::with_id(1),
                Voxel::with_id(2),
                Voxel::with_id(3),
                Voxel::with_id(4),
                Voxel::with_id(5),
                Voxel::with_id(6),
                Voxel::with_id(7),
                Voxel::with_id(7),
            ]),
            Voxel::with_id(7)
        );
    }
}
