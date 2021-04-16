use std::sync::Arc;
use dust_core::svo::alloc::BlockAllocator;
use dust_core::svo::octree::supertree::OctreeLoader;
use dust_core::Voxel;
use dust_core::svo::ArenaAllocator;
use dust_core::Octree;

pub struct Loader {
    block_allocator: Arc<dyn BlockAllocator>
}

impl Loader {
    pub fn new(block_allocator: Arc<dyn BlockAllocator>) -> Self {
        Loader {
            block_allocator
        }
    }
}
impl OctreeLoader<Voxel> for Loader {
    fn load_region_with_lod(&self, region_x: u64, region_y: u64, region_z: u64, distance_to_center: u8) -> Option<Octree> {
        if distance_to_center != 0 {
            return None;
        }
        if region_y != 3 {
            return None;
        }
        if region_x != 3 || region_z != 3 {
            if region_x != 3 || region_z != 4 {
                return None;
            }
        }
        let arena_allocator = ArenaAllocator::new(self.block_allocator.clone());
        let mut octree = Octree::new(arena_allocator);

        let file =
            std::fs::File::open(format!("{}/r.{}.{}.mca", "./assets/region", region_x, region_z))
                .unwrap();
        let region_x = region_x + 7;
        let region_y = region_z + 6;
        let mut region = fastanvil::Region::new(file);

        region
            .for_each_chunk(|chunk_x, chunk_z, chunk_data| {
                let mut mutator = octree.get_random_mutator();
                let chunk: fastanvil::Chunk =
                    fastnbt::de::from_bytes(chunk_data.as_slice()).unwrap();

                if let Some(sections) = chunk.level.sections {
                    for section in sections {
                        if section.palette.is_none() {
                            continue;
                        }
                        let palette = section.palette.unwrap();
                        if let Some(block_states) = section.block_states {
                            let bits_per_item = (block_states.0.len() * 8) / 4096;
                            let mut buff: [u16; 4096] = [0; 4096];
                            block_states.unpack_into(bits_per_item, &mut buff);
                            for (i, indice) in buff.iter().enumerate() {
                                let indice = *indice;
                                let block = &palette[indice as usize];
                                let x = (i & 0xF) as u32;
                                let z = ((i >> 4) & 0xF) as u32;
                                let y = (i >> 8) as u32;

                                let y = y + section.y as u32 * 16;
                                assert_eq!(i >> 12, 0);
                                let voxel = match block.name {
                                    "minecraft:air" => continue,
                                    "minecraft:cave_air" => continue,
                                    "minecraft:grass" => continue,
                                    "minecraft:tall_grass" => continue,
                                    _ => Voxel::with_id(1),
                                };
                                mutator.set(
                                    x + chunk_x as u32 * 16,
                                    y,
                                    z + chunk_z as u32 * 16,
                                    512,
                                    voxel,
                                );
                            }
                        }
                    }
                }
            })
            .unwrap();
        println!("Region loaded: {} {}", region_x, region_y);
        Some(octree)
    }

    fn unload_region(&self, x: u64, y: u64, z: u64, octree: Octree) {
        todo!()
    }
}