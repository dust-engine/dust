use crate::bounds::Bounds;
use crate::dir::{Corner, Edge};
use crate::mesher::stack::StackAllocator;
use crate::mesher::surface::Surface;
use crate::mesher::Mesh;
use crate::octree::accessor::tree::NodeRef;
use crate::octree::Octree;
use crate::Voxel;
use glam::{Vec2, Vec3};
use std::fmt::Debug;

pub struct MarchingCubeMeshBuilder<T> {
    vertices: Vec<Vec3>,
    indices: Vec<u32>,
    uvs: Vec<Vec2>,
    normals: Vec<Vec3>,
    lod: u8,
    size: f32,
    current: u32,
    stack_allocator: StackAllocator<T>,
}

// XZ Plane
//       5-------------------4
//      /                   /
//     /                   /
//    /                   /
//   6-------------------7

// XY Plane
//   1----------0
//   |          |
//   |          |
//   2----------3

// YZ Plane
//       2----------6
//      /|         /|
//     3-|--------7 |
//     | |        | |    <-----
//     | 0--------|-4
//     |/         |/
//     1----------5

#[macro_use]
mod foo {
    macro_rules! build_surface {
        (XY; $self: ident, $lod: expr, $min_surface: expr, $max_surface: expr, $node: expr; $color: expr) => {
            build_surface!($self, $lod, $min_surface, $max_surface, $node; 0 1 2 false true; $color);
        };
        (XZ; $self: ident, $lod: expr, $min_surface: expr, $max_surface: expr, $node: expr; $color: expr) => {
            build_surface!($self, $lod, $min_surface, $max_surface, $node; 0 2 1 true true; $color);
        };
        (YZ; $self: ident, $lod: expr, $min_surface: expr, $max_surface: expr, $node: expr; $color: expr) => {
            build_surface!($self, $lod, $min_surface, $max_surface, $node; 2 1 0 false false; $color);
        };
        ($self: ident, $lod: expr, $min_surface: expr, $max_surface: expr, $node: expr; $x: tt $y: tt $z: tt $skipa: literal $skipb: literal; $color: expr) => {
            let size = 1 << $lod;
            let cell_width = Bounds::MAX_WIDTH >> $self.lod;
            let bounds_offset = $node.get_bounds().width/2 - cell_width;
            let mut rows = [$min_surface.get_first_row(), $max_surface.get_first_row()];
            for a in 0..size-1 {
                let next_rows = [rows[0].next(), rows[1].next()];
                if $skipa && a == size/2 - 1 {
                    rows = next_rows;
                    continue;
                }
                for b in 0..size-1 {
                    if $skipb && b == size/2 - 1 { continue }
                    let mut edge_index: u8 = 0;
                    for dir in Corner::all() {
                        let offset = dir.position_offset();

                        let row = [&rows, &next_rows][offset.$y as usize][offset.$z as usize];
                        let surface = [$min_surface, $max_surface][offset.$z as usize];

                        let data = unsafe {
                            // We know this is now initialized, because build_recursive was run.
                            surface.get_in_row(&mut $self.stack_allocator, row, b + offset.$x as usize).assume_init()
                        };
                        edge_index = edge_index << 1;
                        if data != Default::default() {
                            edge_index |= 1;
                        }
                    }

                    let bounds = $node.get_bounds();
                    let params: (u32, u32, u32) = (b as u32 * cell_width, a as u32 * cell_width, bounds_offset);
                    let bounds = Bounds {
                        x: bounds.x + params.$x,
                        y: bounds.y + params.$y,
                        z: bounds.z + params.$z,
                        width: cell_width * 2
                    };
                    if edge_index != 0 && edge_index != 255 {

                    $self.draw_edge_index(edge_index, &bounds, $color);
                    }
                }
                rows = next_rows;
            }
        }
    }
}

impl<T: Voxel + Debug> MarchingCubeMeshBuilder<T> {
    pub fn new(size: f32, lod: u8) -> MarchingCubeMeshBuilder<T> {
        let grid_size: usize = 1 << lod;
        MarchingCubeMeshBuilder {
            vertices: Vec::new(),
            indices: Vec::new(),
            uvs: Vec::new(),
            normals: Vec::new(),
            lod,
            size,
            current: 0,
            // 8n^2 can be proven to be the max amount of memory needed
            stack_allocator: StackAllocator::new(grid_size * grid_size * 12),
        }
    }
    fn add_triangle(&mut self, edges: [Edge; 3], bounds: &Bounds, color: Vec3) {
        for edge in edges.iter() {
            let (v1, v2) = edge.vertices();
            let node1 = bounds.half(v1).center();
            let node2 = bounds.half(v2).center();
            let mut pos = (node1 + node2) * (self.size * 0.5);
            //pos.y -= 6.0;
            //pos.x = self.size-pos.x;
            pos.z = self.size - pos.z;
            self.vertices.push(pos.into());
            self.normals.push(color);
            self.current += 1;
        }
        // Making faces visible from both sides
        self.indices.push(self.current - 1);
        self.indices.push(self.current - 2);
        self.indices.push(self.current - 3);
    }

    fn draw_edge_index(&mut self, edge_index: u8, bounds: &Bounds, color: Vec3) {
        let edge_table: &[u64; 256] = unsafe {
            let slice: &[u64] =
                std::slice::from_raw_parts(include_bytes!("mc.bin").as_ptr() as *const u64, 256);
            std::mem::transmute(slice.as_ptr())
        };
        let mut edge_bin = edge_table[edge_index as usize];
        for _edge_index in 0..5 {
            let edges = edge_bin & 0xfff;
            if edges == 0xfff {
                break;
            }
            edge_bin = edge_bin >> 12;
            let edge1: Edge = ((edges & 0b1111) as u8).into();
            let edge2: Edge = (((edges >> 4) & 0b1111) as u8).into();
            let edge3: Edge = ((edges >> 8) as u8).into();

            self.add_triangle([edge3, edge2, edge1], bounds, color);
        }
    }

    fn build_recursive(&mut self, node: NodeRef<T>, lod: u8, borders: [Option<Surface>; 6]) {
        let size = 1 << lod;
        if node.is_virtual() {
            // Virtual nodes always have all eight children with the same color.
            // Therefore it could be skipped.
            // But before that, fill the borders.
            let value = node.get();
            for i in &borders {
                if let Some(i) = i {
                    i.fill(&mut self.stack_allocator, value);
                }
            }
            return;
        }

        if lod == 1 {
            let mut edge_index: u8 = 0;
            for corner in Corner::all() {
                let child = node.child(corner).get();
                edge_index = edge_index << 1;
                if child != Default::default() {
                    edge_index |= 1;
                }

                // write into borders
                for (internal, face, quadrant) in &corner.subdivided_surfaces() {
                    if *internal {
                        continue;
                    }
                    if let Some(surface) = borders[*face as usize] {
                        let surface = surface.slice(*quadrant);
                        assert_eq!(surface.size, 1);
                        surface
                            .get_mut(&mut self.stack_allocator, 0, 0)
                            .write(child);
                    }
                }
            }
            self.draw_edge_index(edge_index, node.get_bounds(), Vec3::new(0.0, 0.0, 0.0));
            // Needs to write into borders in here, as well.
        } else {
            // Creating the cross for subnodes
            let surface_xy_minz = Surface::new(self.stack_allocator.allocate(size * size), size); // Back
            let surface_xy_maxz = Surface::new(self.stack_allocator.allocate(size * size), size); // Front
            let surface_yz_minx = Surface::new(self.stack_allocator.allocate(size * size), size); // Left
            let surface_yz_maxx = Surface::new(self.stack_allocator.allocate(size * size), size); // Right
            let surface_xz_miny = Surface::new(self.stack_allocator.allocate(size * size), size); // Bottom
            let surface_xz_maxy = Surface::new(self.stack_allocator.allocate(size * size), size); // Top
            let internal_surfaces: [Surface; 6] = [
                surface_xz_maxy,
                surface_xz_miny,
                surface_yz_minx,
                surface_yz_maxx,
                surface_xy_maxz,
                surface_xy_minz,
            ];

            for corner in Corner::all() {
                let new_borders: [Option<Surface>; 6] =
                    corner
                        .subdivided_surfaces()
                        .map(|(internal, face, quadrant)| {
                            if internal {
                                Some(internal_surfaces[face as usize].slice(quadrant))
                            } else {
                                borders[face as usize].map(|surface| surface.slice(quadrant))
                            }
                        });
                self.build_recursive(node.child(corner), lod - 1, new_borders);
            }

            // At this point, all surfaces should be filled.
            // Run Marching cube algorithms on the 3 surfaces
            let _cell_width = Bounds::MAX_WIDTH >> self.lod;

            build_surface!(XY; self, lod, &surface_xy_maxz, &surface_xy_minz, node; Vec3::new(1.0, 0.0, 0.0));
            build_surface!(XZ; self, lod, &surface_xz_miny, &surface_xz_maxy, node; Vec3::new(0.0, 1.0, 0.0));
            build_surface!(YZ; self, lod, &surface_yz_minx, &surface_yz_maxx, node; Vec3::new(0.0, 0.0, 1.0));

            unsafe {
                self.stack_allocator.deallocate_size(size * size * 6);
            }
        }
    }

    pub fn build(mut self, octree: &Octree<T>) -> Mesh {
        self.build_recursive(
            octree.get_tree_accessor(),
            self.lod,
            [None, None, None, None, None, None],
        );
        Mesh {
            vertices: self.vertices,
            indices: self.indices,
            normals: self.normals,
            uvs: self.uvs,
        }
    }
}

#[cfg(untested)]
mod tests {
    use super::*;
    extern crate test;
    use test::Bencher;

    #[bench]
    fn bench_sphere(b: &mut Bencher) {
        let lod = 6;
        let octree: Octree<u16> = Octree::from_signed_distance_field(
            |l: glam::Vec3| 0.4 - l.distance(Vec3::new(0.5, 0.5, 0.5)),
            1,
            lod,
        );
        b.iter(|| {
            let mut builder: MarchingCubeMeshBuilder<u16> = MarchingCubeMeshBuilder::new(3.0, lod);
            let mesha = builder.build(&octree);
            mesha
        })
    }

    #[bench]
    fn bench_inf_norm(b: &mut Bencher) {
        let lod = 6;
        let octree: Octree<u16> = Octree::from_signed_distance_field(
            |l: glam::Vec3| {
                let l = l - Vec3::new(0.5, 0.5, 0.5);
                0.4 - l.x.abs().max(l.y.abs()).max(l.z.abs())
            },
            1,
            lod,
        );
        b.iter(|| {
            let mut builder: MarchingCubeMeshBuilder<u16> = MarchingCubeMeshBuilder::new(3.0, lod);
            let mesha = builder.build(&octree);
            mesha
        })
    }
}
