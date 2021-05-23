use itertools::Itertools;
use std::fmt::Debug;
use std::ops::Sub;

/// A point on the surface of a [Topology].
/// Note that the `Sub` implementation should compute the distance between the points.
trait SurfacePoint: Sub<Output = f64> + Sized + Copy + Debug {
    fn weighted_average(points: &[(f64, Self)]) -> Self;

    fn average(points: &[Self]) -> Self {
        Self::weighted_average(&points.iter().map(|p| (1., *p)).collect::<Vec<_>>()[..])
    }

    fn triangle_area(triangle: [Self; 3]) -> f64;

    fn triangle_centroid(triangle: [Self; 3]) -> Self {
        // TODO: Figure out whether this is valid.
        Self::average(&triangle)
    }

    fn centroid(polygon: &[Self]) -> Self {
        let center = Self::average(polygon);
        Self::weighted_average(
            &polygon
                .iter()
                .circular_tuple_windows()
                .map(|(x, y)| {
                    let triangle = [*x, *y, center];
                    (
                        Self::triangle_area(triangle),
                        Self::triangle_centroid(triangle),
                    )
                })
                .collect::<Vec<_>>()[..],
        )
    }
}

trait Topology {
    type Point: SurfacePoint;

    fn random_point(&self) -> Self::Point;
}
