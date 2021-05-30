use crate::partition::Partition;
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::Add;
use std::ops::Div;
use std::ops::Mul;
use std::ops::Sub;

/// The topology of a surface.
pub trait Topology: Sized {
    type Point: SurfacePoint<Self::Vector>;
    type Vector: SurfaceVector<Self::Point>;
    type Partitioner: Partitioner<Self>;
}

/// A point on the surface of a [Topology].
pub trait SurfacePoint<Vector: SurfaceVector<Self>>:
    Add<Vector, Output = Self> + Sub<Output = Vector> + Sized + Copy + Debug
{
}

/// A vector on the surface of a [Topology].
pub trait SurfaceVector<Point: SurfacePoint<Self>>:
    Add<Point, Output = Point>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<f64, Output = Self>
    + Div<f64, Output = Self>
    + Sized
    + Copy
    + Debug
{
    fn magnitude(self) -> f64;
    fn normalized(self) -> Self {
        self / self.magnitude()
    }
}

/// A method of generating a partition from a topology.
pub trait Partitioner<Top: Topology> {
    fn partition(&self, top: Top, seed: impl Hash) -> Partition<Top>;
}

/* Implemented Topologies */
pub mod disk;
