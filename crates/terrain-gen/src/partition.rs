use crate::topology::Topology;
use smallvec::SmallVec;

#[derive(Clone, Debug)]
pub struct CellVec<Top: Topology> {
    pub position: Vec<Top::Point>,
    pub size: Vec<f64>,
    pub boundary: Vec<SmallVec<[u32; 8]>>,
    pub connections: Vec<SmallVec<[u32; 8]>>,
}

#[derive(Clone, Debug)]
pub struct Partition<Top: Topology> {
    pub cells: CellVec<Top>,
    pub boundary_points: Vec<Top::Point>,
}
