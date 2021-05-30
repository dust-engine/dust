use dust_terrain_gen::topology::Partitioner;
use dust_terrain_gen::*;
use itertools::Itertools;
use macroquad::prelude::*;
use topology::disk::*;

#[macroquad::main("test")]
async fn main() {
    let disk = Disk::new(1.0);
    let size = 2.0;
    let partition = DiskPartitioner::new(100, 5).partition(disk, 0);
    loop {
        let aspect = screen_height() / screen_width();
        set_camera(&Camera2D::from_display_rect(Rect::new(
            -size,
            -size * aspect,
            2.0 * size,
            2.0 * size * aspect,
        )));
        clear_background(BLACK);

        for pos in &partition.cells.position {
            draw_circle(pos.x as f32, pos.y as f32, 0.01, WHITE);
        }

        for boundary in &partition.cells.boundary {
            for ids in boundary.iter().cloned().circular_tuple_windows::<(_, _)>() {
                let a = partition.boundary_points[ids.0 as usize];
                let b = partition.boundary_points[ids.1 as usize];
                draw_line(a.x as f32, a.y as f32, b.x as f32, b.y as f32, 0.01, WHITE);
            }
        }

        next_frame().await
    }
}
