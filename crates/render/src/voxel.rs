use svo::alloc::ArenaAllocated;

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

        let mut count: u8 = 1;
        let mut max_count: u8 = 0;
        let mut max_element: Self = Voxel(0);
        let mut last_element: Self = arr[0];
        for i in arr.iter().skip(1) {
            if *i != last_element {
                if count > max_count {
                    max_count = count;
                    max_element = last_element;
                }
                count = 0;
                last_element = *i;
            }
            count += 1;
        }
        if count > max_count {
            max_element = last_element;
        }
        max_element
    }
}
