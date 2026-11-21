//! Emitter tables: the emissive voxel faces of one model, laid out for light
//! sampling.
//!
//! A model's builder (in `dust_vox`) finds its emissive faces, and
//! [`EmitterTable::build`] groups them into clusters, one per 32³-voxel region
//! of the model and face direction. A cluster keeps the bounds of its faces and
//! their total power, so a shading point can weigh every cluster by power over
//! distance and drop the ones facing away from it, and an alias table over its
//! faces, so it can then pick a face in proportion to its power in O(1).
//!
//! Each table is one GPU buffer, read by device address:
//!
//! ```text
//! EmitterTableHeader                  16 B
//! [EmitterCluster; cluster_count]     48 B each
//! [EmitterFace; face_count]           16 B each, grouped by cluster
//! ```
//!
//! Positions are in the model's object space; directions are numbered as
//! SHARC's face ids (`SharcParams::cellKey`): +x, -x, +y, -y, +z, -z.
//!
//! Power here is luminance times area, without the π of a Lambertian emitter:
//! sampling only uses ratios of it. A face's probability density over area is
//! then `luminance(radiance) / power`, the same for every point of every face.

use bevy::math::{UVec3, Vec3};
use bytemuck::{Pod, Zeroable};

/// Voxels per side of a region, as a power of two: faces are clustered by 32³
/// regions, the size of the voxel tree's second level.
pub const REGION_LOG2: u32 = 5;

/// Rec. 709 luminance, the weights the shaders use.
pub const LUMINANCE: Vec3 = Vec3::new(0.2126, 0.7152, 0.0722);

/// One emissive voxel face, as a model's builder finds it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmissiveFace {
    /// The voxel's coordinate in the model, each component below 256.
    pub coords: UVec3,
    /// The way the face points: 0..6 for +x, -x, +y, -y, +z, -z.
    pub direction: u8,
    /// The radiance it emits, as the shaders compute it: the palette color,
    /// still sRGB-encoded, times the entry's emission.
    pub radiance: Vec3,
}

/// The start of an emitter table.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, PartialEq)]
pub struct EmitterTableHeader {
    pub cluster_count: u32,
    pub face_count: u32,
    /// The sum of the clusters' power.
    pub power: f32,
    /// Object-space size of a voxel. A face's area is its square.
    pub unit_size: f32,
}

/// The faces of one region that point one way.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, PartialEq)]
pub struct EmitterCluster {
    /// Object-space bounds of the faces themselves (flat along the direction's
    /// axis when they all lie in one plane). A point sees none of the faces
    /// unless it lies beyond the bound's nearest face plane: above `min` on
    /// the axis for a positive direction, below `max` for a negative one.
    pub min: [f32; 3],
    /// Index of the cluster's first face in the table's face array.
    pub face_offset: u32,
    pub max: [f32; 3],
    pub face_count: u32,
    /// The sum of the faces' power.
    pub power: f32,
    /// 0..6 for +x, -x, +y, -y, +z, -z.
    pub direction: u32,
    pub _pad: [u32; 2],
}

/// One emissive face and its alias-table entry.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, PartialEq)]
pub struct EmitterFace {
    /// The voxel's coordinate and the face's direction: `x | y << 8 | z << 16 |
    /// direction << 24`.
    pub packed: u32,
    /// The emitted radiance, RGB9E5 ([`encode_rgb9e5`]).
    pub radiance: u32,
    /// Alias table: to pick a face of a cluster, draw `i` uniformly among its
    /// faces and `u` in [0, 1); take face `i` if `u < alias_threshold`, and
    /// face `alias` (an index within the cluster) otherwise.
    pub alias_threshold: f32,
    pub alias: u32,
}

impl EmitterFace {
    pub fn coords(&self) -> UVec3 {
        UVec3::new(
            self.packed & 0xFF,
            (self.packed >> 8) & 0xFF,
            (self.packed >> 16) & 0xFF,
        )
    }
    pub fn direction(&self) -> u8 {
        (self.packed >> 24) as u8
    }
}

/// The emitter table of one model, on the CPU.
#[derive(Clone, Debug, PartialEq)]
pub struct EmitterTable {
    pub header: EmitterTableHeader,
    pub clusters: Vec<EmitterCluster>,
    pub faces: Vec<EmitterFace>,
}

impl EmitterTable {
    /// Groups `faces` into clusters and builds each cluster's alias table.
    /// Faces that emit no luminance are left out. The order of `faces` doesn't
    /// matter: the table depends only on the set.
    pub fn build(unit_size: f32, faces: &[EmissiveFace]) -> Self {
        let area = unit_size * unit_size;
        let mut faces: Vec<(EmissiveFace, f32)> = faces
            .iter()
            .filter_map(|&face| {
                debug_assert!(face.coords.cmplt(UVec3::splat(256)).all());
                debug_assert!(face.direction < 6);
                let power = face.radiance.dot(LUMINANCE) * area;
                (power > 0.0).then_some((face, power))
            })
            .collect();
        faces.sort_by_key(|(face, _)| sort_key(face));

        let mut clusters = Vec::new();
        let mut table_faces = Vec::with_capacity(faces.len());
        let mut total_power = 0.0;
        for group in faces.chunk_by(|(a, _), (b, _)| cluster_key(a) == cluster_key(b)) {
            let direction = group[0].0.direction;
            let mut min = Vec3::MAX;
            let mut max = Vec3::MIN;
            for (face, _) in group {
                let (face_min, face_max) = face_bounds(face.coords, face.direction);
                min = min.min(face_min);
                max = max.max(face_max);
            }
            let weights: Vec<f32> = group.iter().map(|&(_, power)| power).collect();
            let power = weights.iter().sum::<f32>();
            total_power += power;
            clusters.push(EmitterCluster {
                min: (min * unit_size).to_array(),
                face_offset: table_faces.len() as u32,
                max: (max * unit_size).to_array(),
                face_count: group.len() as u32,
                power,
                direction: direction as u32,
                _pad: [0; 2],
            });
            let alias = alias_table(&weights);
            table_faces.extend(group.iter().zip(alias).map(|(&(face, _), (threshold, alias))| {
                EmitterFace {
                    packed: face.coords.x
                        | face.coords.y << 8
                        | face.coords.z << 16
                        | (face.direction as u32) << 24,
                    radiance: encode_rgb9e5(face.radiance),
                    alias_threshold: threshold,
                    alias,
                }
            }));
        }
        Self {
            header: EmitterTableHeader {
                cluster_count: clusters.len() as u32,
                face_count: table_faces.len() as u32,
                power: total_power,
                unit_size,
            },
            clusters,
            faces: table_faces,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.faces.is_empty()
    }

    /// The size of the table's GPU buffer, in bytes.
    pub fn byte_size(&self) -> usize {
        size_of::<EmitterTableHeader>()
            + self.clusters.len() * size_of::<EmitterCluster>()
            + self.faces.len() * size_of::<EmitterFace>()
    }

    /// Writes the table's GPU layout into `dst`, which must be at least
    /// [`byte_size`](Self::byte_size) bytes.
    pub fn write_to(&self, dst: &mut [u8]) {
        let parts: [&[u8]; 3] = [
            bytemuck::bytes_of(&self.header),
            bytemuck::cast_slice(&self.clusters),
            bytemuck::cast_slice(&self.faces),
        ];
        let mut offset = 0;
        for part in parts {
            dst[offset..offset + part.len()].copy_from_slice(part);
            offset += part.len();
        }
    }
}

/// The region a face belongs to and its direction: faces sharing these share a
/// cluster.
fn cluster_key(face: &EmissiveFace) -> (UVec3, u8) {
    (face.coords >> REGION_LOG2, face.direction)
}

/// Orders faces by cluster, regions in the tree's order (x-major, z fastest),
/// then by coordinate within the cluster, so the table doesn't depend on the
/// order the faces were found in.
fn sort_key(face: &EmissiveFace) -> (u32, u8, u32) {
    let pack = |c: UVec3| c.x << 16 | c.y << 8 | c.z;
    (pack(face.coords >> REGION_LOG2), face.direction, pack(face.coords))
}

/// The bounds, in voxels, of the face of the voxel at `coords` that points in
/// `direction`: the voxel's cube, flattened onto that side.
fn face_bounds(coords: UVec3, direction: u8) -> (Vec3, Vec3) {
    let min = coords.as_vec3();
    let mut face_min = min;
    let mut face_max = min + 1.0;
    let axis = (direction / 2) as usize;
    let plane = if direction % 2 == 0 { min[axis] + 1.0 } else { min[axis] };
    face_min[axis] = plane;
    face_max[axis] = plane;
    (face_min, face_max)
}

/// Builds an alias table over `weights` by Vose's method: entry `i` keeps
/// `i` with probability `threshold` and gives the rest to `alias`, so drawing
/// an entry uniformly and then choosing between it and its alias picks `i`
/// with probability `weights[i] / sum`.
fn alias_table(weights: &[f32]) -> Vec<(f32, u32)> {
    let n = weights.len();
    let sum: f64 = weights.iter().map(|&w| w as f64).sum();
    // Each entry's weight, scaled so the average is 1.
    let mut scaled: Vec<f64> = weights.iter().map(|&w| w as f64 * n as f64 / sum).collect();
    let mut table: Vec<(f32, u32)> = (0..n as u32).map(|i| (1.0, i)).collect();
    let (mut small, mut large): (Vec<usize>, Vec<usize>) = (0..n).partition(|&i| scaled[i] < 1.0);
    while let (Some(&s), Some(&l)) = (small.last(), large.last()) {
        small.pop();
        table[s] = (scaled[s] as f32, l as u32);
        // `l` gives `s` the probability `s` lacks.
        scaled[l] -= 1.0 - scaled[s];
        if scaled[l] < 1.0 {
            large.pop();
            small.push(l);
        }
    }
    // Whatever is left is 1 up to rounding, and keeps itself: (1.0, i).
    table
}

/// Packs a radiance into the shared-exponent format of
/// `VK_FORMAT_E5B9G9R9_UFLOAT_PACK32`: a 9-bit mantissa per channel and one
/// 5-bit exponent, `r | g << 9 | b << 18 | exponent << 27`. Negative and NaN
/// channels become 0 and channels above 65408 clamp to it.
pub fn encode_rgb9e5(rgb: Vec3) -> u32 {
    const MANTISSA_BITS: i32 = 9;
    const BIAS: i32 = 15;
    const MAX: f32 = 65408.0; // (2^9 - 1) / 2^9 * 2^(31 - 15)
    let rgb = Vec3::new(
        clamp_channel(rgb.x, MAX),
        clamp_channel(rgb.y, MAX),
        clamp_channel(rgb.z, MAX),
    );
    let max = rgb.max_element();
    if max == 0.0 {
        return 0;
    }
    let mut exponent = (max.log2().floor() as i32).max(-BIAS - 1) + 1 + BIAS;
    let scale = |exponent: i32| 2f32.powi(exponent - BIAS - MANTISSA_BITS);
    if (max / scale(exponent) + 0.5).floor() as u32 == 1 << MANTISSA_BITS {
        exponent += 1;
    }
    let mantissa = |c: f32| ((c / scale(exponent) + 0.5).floor() as u32).min(511);
    mantissa(rgb.x) | mantissa(rgb.y) << 9 | mantissa(rgb.z) << 18 | (exponent as u32) << 27
}

fn clamp_channel(c: f32, max: f32) -> f32 {
    if c > 0.0 { c.min(max) } else { 0.0 }
}

/// Unpacks [`encode_rgb9e5`].
pub fn decode_rgb9e5(packed: u32) -> Vec3 {
    let exponent = (packed >> 27) as i32;
    let scale = 2f32.powi(exponent - 15 - 9);
    Vec3::new(
        (packed & 0x1FF) as f32,
        ((packed >> 9) & 0x1FF) as f32,
        ((packed >> 18) & 0x1FF) as f32,
    ) * scale
}

#[cfg(test)]
mod tests {
    use super::*;

    fn face(x: u32, y: u32, z: u32, direction: u8, radiance: f32) -> EmissiveFace {
        EmissiveFace {
            coords: UVec3::new(x, y, z),
            direction,
            radiance: Vec3::splat(radiance),
        }
    }

    /// The probability the alias table of `faces` picks each of them.
    fn alias_probabilities(faces: &[EmitterFace]) -> Vec<f64> {
        let n = faces.len() as f64;
        let mut p = vec![0.0; faces.len()];
        for (i, face) in faces.iter().enumerate() {
            p[i] += face.alias_threshold as f64 / n;
            p[face.alias as usize] += (1.0 - face.alias_threshold as f64) / n;
        }
        p
    }

    #[test]
    fn alias_tables_reproduce_power() {
        // Weights spanning two orders of magnitude, in one cluster. The faces
        // are already in table order (ascending x).
        let radiance = |i: u32| [0.4, 41.0, 3.0, 17.0, 0.9][i as usize % 5];
        let faces: Vec<_> = (0..20).map(|i| face(i, 0, 0, 2, radiance(i))).collect();
        let table = EmitterTable::build(1.0 / 16.0, &faces);
        assert_eq!(table.clusters.len(), 1);
        let cluster = table.clusters[0];
        // A gray radiance's luminance is its value: the weights sum to 1.
        let total: f64 = (0..20).map(|i| radiance(i) as f64).sum();
        for (i, p) in alias_probabilities(&table.faces).iter().enumerate() {
            let expected = radiance(i as u32) as f64 / total;
            assert!((p - expected).abs() < 1e-6, "face {i}: {p} vs {expected}");
        }
        assert!((cluster.power as f64 - total / 256.0).abs() < 1e-5);
        for face in &table.faces {
            assert!((0.0..=1.0).contains(&face.alias_threshold));
            assert!(face.alias < cluster.face_count);
        }
    }

    #[test]
    fn clusters_split_by_region_and_direction() {
        let faces = [
            face(0, 0, 0, 0, 1.0),
            face(31, 31, 31, 0, 1.0),
            face(0, 0, 0, 2, 1.0),  // same region, other direction
            face(32, 0, 0, 0, 1.0), // next region along x
            face(5, 5, 5, 0, 0.0),  // emits nothing: left out
        ];
        let table = EmitterTable::build(0.5, &faces);
        assert_eq!(table.header.face_count, 4);
        assert_eq!(table.header.cluster_count, 3);
        let first = table.clusters[0];
        assert_eq!((first.direction, first.face_count, first.face_offset), (0, 2, 0));
        // +x faces sit on the voxels' x + 1 planes.
        assert_eq!(first.min, [0.5, 0.0, 0.0]);
        assert_eq!(first.max, [16.0, 16.0, 16.0]);
        assert_eq!(table.clusters[1].direction, 2);
        assert_eq!(table.clusters[2].min, [16.5, 0.0, 0.0]);

        let total: f32 = table.clusters.iter().map(|c| c.power).sum();
        assert_eq!(table.header.power, total);
        assert!((total - 4.0 * 0.25).abs() < 1e-6);

        // Every face lies inside its cluster's bounds.
        for cluster in &table.clusters {
            let range = cluster.face_offset as usize..(cluster.face_offset + cluster.face_count) as usize;
            for f in &table.faces[range] {
                let (min, max) = face_bounds(f.coords(), f.direction());
                assert!((min * 0.5).cmpge(Vec3::from(cluster.min)).all());
                assert!((max * 0.5).cmple(Vec3::from(cluster.max)).all());
                assert_eq!(f.direction() as u32, cluster.direction);
            }
        }
    }

    #[test]
    fn table_ignores_face_order() {
        let faces: Vec<_> = (0..50).map(|i| face(i * 3 % 64, i % 7, i, (i % 6) as u8, 1.0 + i as f32)).collect();
        let mut reversed = faces.clone();
        reversed.reverse();
        assert_eq!(EmitterTable::build(1.0, &faces), EmitterTable::build(1.0, &reversed));
    }

    #[test]
    fn gpu_layout() {
        assert_eq!(size_of::<EmitterTableHeader>(), 16);
        assert_eq!(size_of::<EmitterCluster>(), 48);
        assert_eq!(size_of::<EmitterFace>(), 16);
        let table = EmitterTable::build(1.0, &[face(1, 2, 3, 5, 2.0)]);
        let mut bytes = vec![0u8; table.byte_size()];
        table.write_to(&mut bytes);
        assert_eq!(bytes.len(), 16 + 48 + 16);
        let face: &EmitterFace = bytemuck::from_bytes(&bytes[64..80]);
        assert_eq!((face.coords(), face.direction()), (UVec3::new(1, 2, 3), 5));
    }

    #[test]
    fn rgb9e5_round_trip() {
        for rgb in [
            Vec3::ZERO,
            Vec3::new(1.0, 0.5, 0.25),
            Vec3::new(48.0, 3.6, 0.0),
            Vec3::new(1e-3, 2e-3, 0.0),
            Vec3::splat(65408.0),
        ] {
            let decoded = decode_rgb9e5(encode_rgb9e5(rgb));
            let tolerance = rgb.max_element() / 512.0;
            assert!((decoded - rgb).abs().max_element() <= tolerance, "{rgb} -> {decoded}");
        }
        assert_eq!(decode_rgb9e5(encode_rgb9e5(Vec3::splat(1e9))), Vec3::splat(65408.0));
        assert_eq!(decode_rgb9e5(encode_rgb9e5(Vec3::new(-1.0, f32::NAN, 2.0))).x, 0.0);
    }
}
