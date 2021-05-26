use crate::partition::Partition;
use itertools::Itertools;

use std::fmt::Debug;
use std::ops::Add;
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
    /// Computes the weighted average of the points. Note that the first element of the tuple does not necessarily sum up to one.
    fn weighted_average(points: impl Iterator<Item = (f64, Self)>) -> Self;

    fn average(points: impl Iterator<Item = Self>) -> Self {
        Self::weighted_average(points.map(|p| (1., p)))
    }

    fn triangle_area(triangle: [Self; 3]) -> f64;

    fn triangle_centroid(triangle: [Self; 3]) -> Self {
        // TODO: Figure out whether this is valid.
        Self::average(triangle.iter().cloned())
    }

    fn centroid(polygon: &[Self]) -> Self {
        let center = Self::average(polygon.iter().cloned());
        Self::weighted_average(
            polygon
                .iter()
                .circular_tuple_windows()
                .map(|(x, y)| {
                    let triangle = [*x, *y, center];
                    (
                        Self::triangle_area(triangle),
                        Self::triangle_centroid(triangle),
                    )
                }),
        )
    }
}

/// A vector on the surface of a [Topology].
pub trait SurfaceVector<Point: SurfacePoint<Self>>:
    Add<Point, Output = Point> + Add<Output = Self> + Sub<Output = Self> + Sized + Copy + Debug
{
    fn magnitude(self) -> f64;
}

/// A method of generating a partition from a topology.
pub trait Partitioner<Top: Topology> {
    fn generate_partition(&self, top: Top) -> Partition<Top>;
}
