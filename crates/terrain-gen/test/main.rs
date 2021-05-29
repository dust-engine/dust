use dust_terrain_gen::topology::Partitioner;
use dust_terrain_gen::*;
use macroquad::prelude::*;
use topology::disk::*;

#[macroquad::main("test")]
async fn main() {
    let disk = Disk::new(1.0);
    let partitioner = DiskPartitioner::new(10, 0);
    println!("{:?}", partitioner.partition(disk));
    loop {
        set_camera(&Camera2D::from_display_rect(Rect::new(
            -1.0, -1.0, 1.0, 1.0,
        )));
        clear_background(RED);

        draw_line(-1.0, -1.0, 1.0, 1.0, 0.35, BLUE);

        next_frame().await
    }
}
