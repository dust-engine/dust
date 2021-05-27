use crate::topology::*;
use derive_new::*;
use glam::DVec2;
use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
use std::f64::consts::PI;
use voronoice::VoronoiBuilder;

#[derive(new)]
pub struct Disk {
    radius: f64,
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

#[derive(new)]
pub struct DiskPartitioner {
    num_points: u32,
    relaxation_iterations: u32,
}

impl Partitioner<Disk> for DiskPartitioner {
    fn generate_partition(&self, disk: Disk) -> Partition<Disk> {
        let mut rng = SmallRng::from_entropy();
        let points = (0..self.num_points)
            .map(|_| {
                let point = disk.random_point(&mut rng);
                voronoice::Point {
                    x: point.x,
                    y: point.y,
                }
            })
            .collect::<Vec<_>>();
        let voronoi_diagram = VoronoiBuilder::default()
            .set_sites(points)
            .set_lloyd_relaxation_iterations(self.relaxation_iterations as usize)
            .build()
            .unwrap();
        todo!()
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
