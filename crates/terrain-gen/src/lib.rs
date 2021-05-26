use itertools::Itertools;
use std::fmt::Debug;
use std::ops::Sub;

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
struct Triangle<P> {
    verticies: [P; 3],
}

/// A point on the surface of a [Topology].
/// Note that the `Sub` implementation should compute the distance between the points.
trait SurfacePoint: Sub<Output = f64> + Sized + Copy + Debug {
    fn weighted_average(points: &[(f64, Self)]) -> Self;

    fn average(points: &[Self]) -> Self {
        Self::weighted_average(&points.iter().map(|p| (1., *p)).collect::<Vec<_>>()[..])
    }

    fn triangle_area(triangle: Triangle<Self>) -> f64;

    fn triangle_centroid(triangle: Triangle<Self>) -> Self {
        // TODO: Figure out whether this is valid.
        Self::average(&triangle.verticies)
    }

    fn centroid(polygon: &[Self]) -> Self {
        let center = Self::average(polygon);
        Self::weighted_average(
            &polygon
                .iter()
                .circular_tuple_windows()
                .map(|(x, y)| {
                    let triangle = Triangle {
                        verticies: [*x, *y, center],
                    };
                    (
                        Self::triangle_area(triangle),
                        Self::triangle_centroid(triangle),
                    )
                })
                .collect::<Vec<_>>()[..],
        )
    }
}

struct Partition<Top: Topology> {
    points: Vec<Top::Point>,
    /* TODO */
}

trait Partitioner<Top: Topology> {
    fn generate_partition(&self, top: Top) -> Partition<Top>;

}

trait Topology: Sized {
    type Point: SurfacePoint;
    type Partitioner: Partitioner<Self>;

    fn random_point(&self) -> Self::Point;
}
