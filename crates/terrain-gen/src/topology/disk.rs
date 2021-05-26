use crate::topology::*;
use derive_new::*;
use glam::DVec2;

#[derive(new)]
pub struct Disk {
    radius: f64,
}

impl Topology for Disk {
    type Point = DVec2;
    type Vector = DVec2;
    type Partitioner = DiskPartitioner;
}

#[derive(new)]
pub struct DiskPartitioner {
    num_points: u32,
}

impl Partitioner<Disk> for DiskPartitioner {
    fn generate_partition(&self, disk: Disk) -> Partition<Disk> {
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
