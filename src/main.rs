use crate::fly_camera::FlyCamera;
use bevy::prelude::*;
use dust_core::{Octree, SunLight, Voxel, CameraProjection};
use dust_render::RaytracerCameraBundle;
use std::io::BufWriter;
use std::ops::DerefMut;

mod fly_camera;
mod fps_counter;

fn main() {
    App::build()
        .insert_resource(bevy::log::LogSettings {
            filter: "wgpu=error".to_string(),
            level: bevy::utils::tracing::Level::DEBUG,
        })
        .insert_resource(bevy::window::WindowDescriptor {
            width: 1920.0,
            height: 1080.0,
            scale_factor_override: Some(1.0),
            title: "Dust Engine".to_string(),
            mode: bevy::window::WindowMode::Windowed,
            ..Default::default()
        })
        .insert_resource(fps_counter::FPSCounter::default())
        .add_plugin(bevy::log::LogPlugin::default())
        .add_plugin(bevy::core::CorePlugin::default())
        .add_plugin(bevy::transform::TransformPlugin::default())
        .add_plugin(bevy::diagnostic::DiagnosticsPlugin::default())
        .add_plugin(bevy::diagnostic::LogDiagnosticsPlugin::default())
        .add_plugin(bevy::input::InputPlugin::default())
        .add_plugin(bevy::window::WindowPlugin::default())
        .add_plugin(bevy::winit::WinitPlugin::default())
        .add_plugin(dust_render::DustPlugin::default())
        .add_plugin(fly_camera::FlyCameraPlugin)
        .add_startup_system(setup.system())
        .add_system(run.system())
        .add_system(fps_counter::fps_counter.system())
        .run();
}

fn setup_from_oct_file(mut commands: Commands, mut octree: ResMut<Octree>) {
    let file = std::fs::File::open("./test.oct").unwrap();
    let mut reader = std::io::BufReader::new(file);
    Octree::read(&mut octree, &mut reader, 10).unwrap();
    //let mut accessor = octree.get_random_mutator();
    //accessor.set(1, 1, 1, 8, Voxel::with_id(1));
    //accessor.set(3, 1, 1, 8, Voxel::with_id(1));
    //accessor.set(1, 1, 0, 8, Voxel::with_id(1));
    //accessor.set(0, 0, 0, 8, Voxel::with_id(1));

    let mut bundle = RaytracerCameraBundle::default();
    bundle.transform.translation = Vec3::new(1.0901, 1.3, 1.0894);
    bundle.transform.look_at(Vec3::new(2.0, 0.5, 2.0), Vec3::Y);
    commands.spawn().insert_bundle(bundle).insert(FlyCamera {
        accel: 1.0,
        max_speed: 0.02,
        sensitivity: 10.0,
        ..Default::default()
    });
}

fn setup(mut commands: Commands, mut octree: ResMut<Octree>) {
    let octree = octree.deref_mut();
    let octree_ptr = octree as *mut Octree;
    let octree: &'static mut Octree = unsafe { &mut *octree_ptr };
        let region_dir = "./assets/region";
        let mut load_region = |region_x: i32, region_y: i32| {
            let file =
                std::fs::File::open(format!("{}/r.{}.{}.mca", region_dir, region_x, region_y))
                    .unwrap();
            // let region_x = region_x + 7;
            // let region_y = region_y + 6;
            let mut region = fastanvil::Region::new(file);

            region
                .for_each_chunk(|chunk_x, chunk_z, chunk_data| {
                    let mut mutator = octree.get_random_mutator();
                    let chunk: fastanvil::Chunk =
                        fastnbt::de::from_bytes(chunk_data.as_slice()).unwrap();

                    if let Some(sections) = chunk.level.sections {
                        for section in sections {
                            if section.palette.is_none() {
                                continue;
                            }
                            let palette = section.palette.unwrap();
                            if let Some(block_states) = section.block_states {
                                let bits_per_item = (block_states.0.len() * 8) / 4096;
                                let mut buff: [u16; 4096] = [0; 4096];
                                block_states.unpack_into(bits_per_item, &mut buff);
                                for (i, indice) in buff.iter().enumerate() {
                                    let indice = *indice;
                                    let block = &palette[indice as usize];
                                    let x = (i & 0xF) as u32;
                                    let z = ((i >> 4) & 0xF) as u32;
                                    let y = (i >> 8) as u32;

                                    let y = y + section.y as u32 * 16;
                                    assert_eq!(i >> 12, 0);
                                    let voxel = match block.name {
                                        "minecraft:air" => continue,
                                        "minecraft:cave_air" => continue,
                                        "minecraft:grass" => continue,
                                        "minecraft:tall_grass" => continue,
                                        "minecraft:stone" => Voxel::with_id(1),
                                        "minecraft:bedrock" => Voxel::with_id(1),
                                        "minecraft:dirt" => Voxel::with_id(2),
                                        "minecraft:log" => Voxel::with_id(3),
                                        "minecraft:log2" => Voxel::with_id(3),
                                        "minecraft:planks" => Voxel::with_id(4),
                                        "minecraft:gravel" => Voxel::with_id(5),
                                        "minecraft:white_wool" => Voxel::colored(2, 0),
                                        "minecraft:grey_wool" => Voxel::colored(2, 1),
                                        "minecraft:green_wool" => Voxel::colored(2, 2),
                                        "minecraft:black_wool" => Voxel::colored(2, 3),
                                        "minecraft:blue_wool" => Voxel::colored(2, 4),
                                        "minecraft:brown_wool" => Voxel::colored(2, 5),
                                        "minecraft:cyan_wool" => Voxel::colored(2, 6),
                                        "minecraft:light_blue_wool" => Voxel::colored(2, 7),
                                        "minecraft:light_grey_wool" => Voxel::colored(2, 8),
                                        "minecraft:lime_wool" => Voxel::colored(2, 9),
                                        "minecraft:magenta_wool" => Voxel::colored(2, 10),
                                        "minecraft:orange_wool" => Voxel::colored(2, 11),
                                        "minecraft:pink_wool" => Voxel::colored(2, 12),
                                        "minecraft:purple_wool" => Voxel::colored(2, 13),
                                        "minecraft:red_wool" => Voxel::colored(2, 14),
                                        "minecraft:yellow_wool" => Voxel::colored(2, 15),
                                        "minecraft:grass_block" => Voxel::colored(0, 0),
                                        "minecraft:leaves" => Voxel::colored(1, 0),
                                        "minecraft:leaves2" => Voxel::colored(1, 0),
                                        _ => Voxel::with_id(1),
                                    };
                                    mutator.set(
                                        x + chunk_x as u32 * 16 + region_x as u32 * 512,
                                        y,
                                        z + chunk_z as u32 * 16 + region_y as u32 * 512,
                                        512 * 2, // In order to have dummy root node.
                                        true,
                                    );
                                }
                            }
                        }
                    }
                })
                .unwrap();
            println!("Region loaded: {} {}", region_x, region_y);
        };
        for x in 0..=0 {
            for y in 0..=0 {
                load_region(x, y);
            }
        }
        let file = std::fs::File::create("./test.oct").unwrap();
        let mut bufwriter = BufWriter::new(file);
        octree.write(&mut bufwriter).unwrap();

    let mut bundle = RaytracerCameraBundle::default();
    bundle.transform.translation = Vec3::new(1.0901, 1.3, 1.0894);
    bundle.transform.look_at(Vec3::new(2.0, 0.5, 2.0), Vec3::Y);
    commands.spawn().insert_bundle(bundle).insert(FlyCamera {
        accel: 0.2,
        max_speed: 0.005,
        sensitivity: 10.0,
        ..Default::default()
    });
}

fn run(mut sunlight: ResMut<SunLight>, time: Res<Time>) {
    let (sin, cos) = (time.seconds_since_startup() * 2.0).sin_cos();
    sunlight.dir = Vec3::new(sin as f32 * 10.0, -15.0, cos as f32 * 10.0).normalize();
}
