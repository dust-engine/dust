
/// dox_vox::Voxel to solid materials
pub struct ModelIndexCollector {
    grid: Box<[u8; 256 * 256 * 256]>,
    block_counts: Box<[u32; 64 * 64 * 64]>,
    count: usize,
}
impl ModelIndexCollector {
    pub fn new() -> Self {
        unsafe {
            let grid_ptr = std::alloc::alloc_zeroed(std::alloc::Layout::new::<[u8; 256 * 256 * 256]>());
            let block_counts_ptr = std::alloc::alloc_zeroed(std::alloc::Layout::new::<[u32; 64 * 64 * 64]>());
            
            Self {
                count: 0,
                grid: Box::from_raw(grid_ptr as *mut [u8; 256 * 256 * 256]),
                block_counts: Box::from_raw(block_counts_ptr as *mut [u32; 64 * 64 * 64]),
            }
        }
    }
    pub fn set(&mut self, voxel: dot_vox::Voxel) {
        self.count += 1;
        let block_index = (voxel.x >> 2, voxel.y >> 2, voxel.z >> 2);
        let block_index = block_index.0 as usize + block_index.1 as usize * 64 + block_index.2 as usize * 64 * 64;

        self.block_counts[block_index] += 1;

        let index = (voxel.x & 0b11) | ((voxel.y & 0b11) << 2) | ((voxel.z & 0b11) << 4);
        self.grid[block_index + index as usize] = voxel.i + 1;
    }
}
pub struct ModelIndexCollectorIterator {
    collector: ModelIndexCollector,
    current: usize,
}

impl ModelIndexCollectorIterator {
    pub fn running_sum(&self) -> &[u32; 64*64*64] {
        &self.collector.block_counts
    }
}

impl Iterator for ModelIndexCollectorIterator {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        while self.current < 256 * 256 * 256 {
            let val = self.collector.grid[self.current];
            self.current += 1;
            if val == 0 {
                continue;
            }
            return Some(val);
        }
        None
    }
}
impl ExactSizeIterator for ModelIndexCollectorIterator 
{
    fn len(&self) -> usize {
        self.collector.count
    }
}

impl IntoIterator for ModelIndexCollector {
    type Item = u8;

    type IntoIter = ModelIndexCollectorIterator;

    fn into_iter(mut self) -> Self::IntoIter {
        let mut sum: u32 = 0;
        for i in self.block_counts.iter_mut() {
            *i = sum;
            sum += *i;
        }
        ModelIndexCollectorIterator {
            collector: self,
            current: 0
        }
    }
}
