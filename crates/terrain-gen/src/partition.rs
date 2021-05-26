use crate::topology::Topology;
use smallvec::SmallVec;

#[derive(Clone, Debug)]
pub struct Node<Top: Topology> {
    position: Top::Point,
    size: f64,
    boundary: SmallVec<[u32; 8]>,
    connections: SmallVec<[u32; 8]>,
}

#[derive(Clone, Debug)]
pub struct Partition<Top: Topology> {
    nodes: Vec<Node<Top>>,
    boundary_points: Vec<Top::Point>,
}
