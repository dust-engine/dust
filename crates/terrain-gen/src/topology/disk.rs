use crate::partition::*;
use crate::topology::*;
use ahash::AHasher;
use derive_new::*;
use glam::DVec2;
use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
use smallvec::SmallVec;
use std::f64::consts::PI;
use std::hash::Hash;
use std::hash::Hasher;
use voronoice::NeighborSiteIterator;
use voronoice::VoronoiBuilder;

#[derive(new, Debug, Clone, Copy, PartialEq)]
pub struct Disk {
    pub radius: f64,
}

impl Disk {
    fn random_point(&self, rng: &mut impl Rng) -> DVec2 {
        let r = self.radius * rng.gen::<f64>().sqrt();
        let theta = rng.gen::<f64>() * 2.0 * PI;
        DVec2::new(r * theta.cos(), r * theta.sin())
    }
}

impl Topology for Disk {
    type Point = DVec2;
    type Vector = DVec2;
    type Partitioner = DiskPartitioner;
}

#[derive(new, Debug, Clone, Copy, PartialEq)]
pub struct DiskPartitioner {
    pub num_points: u32,
    pub relaxation_iterations: u32,
}

impl Partitioner<Disk> for DiskPartitioner {
    fn partition(&self, disk: Disk, seed: impl Hash) -> Partition<Disk> {
        let mut hasher = AHasher::new_with_keys(0, 0);
        seed.hash(&mut hasher);
        let mut rng = SmallRng::seed_from_u64(hasher.finish());
        let points = (0..self.num_points)
            .map(|_| {
                let point = disk.random_point(&mut rng);
                voronoice::Point {
                    x: point.x,
                    y: point.y,
                }
            })
            .collect::<Vec<_>>();
        let diagram = VoronoiBuilder::default()
            .set_sites(points)
            .set_lloyd_relaxation_iterations(self.relaxation_iterations as usize)
            .build()
            .unwrap();
        let boundary_points = diagram
            .vertices()
            .iter()
            .map(|p| DVec2::new(p.x, p.y))
            .collect::<Vec<_>>();
        let positions = diagram
            .sites()
            .iter()
            .map(|p| DVec2::new(p.x, p.y))
            .collect::<Vec<_>>();
        let cell_boundaries = diagram
            .cells()
            .iter()
            .map(|ps| ps.iter().map(|p| *p as u32).collect::<SmallVec<[u32; 8]>>())
            .collect::<Vec<_>>();
        let connections = (0..positions.len())
            .map(|i| {
                NeighborSiteIterator::new(&diagram, i)
                    .map(|p| p as u32)
                    .collect::<SmallVec<[u32; 8]>>()
            })
            .collect::<Vec<_>>();
        Partition {
            cells: CellVec {
                position: positions,
                size: vec![],
                boundary: cell_boundaries,
                connections,
            },
            boundary_points,
        }
    }
}

impl SurfacePoint<DVec2> for DVec2 {}

impl SurfaceVector<DVec2> for DVec2 {
    fn magnitude(self) -> f64 {
        self.length()
    }

    fn normalized(self) -> Self {
        self.normalize()
    }
}
