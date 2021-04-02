use crate::fly_camera::FlyCamera;
use bevy::prelude::*;
use bevy_dust::core::{Octree, SunLight, Voxel};
use bevy_dust::RaytracerCameraBundle;

mod fly_camera;

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
        .add_plugin(bevy::log::LogPlugin::default())
        .add_plugin(bevy::core::CorePlugin::default())
        .add_plugin(bevy::transform::TransformPlugin::default())
        .add_plugin(bevy::diagnostic::DiagnosticsPlugin::default())
        .add_plugin(bevy::diagnostic::LogDiagnosticsPlugin::default())
        .add_plugin(bevy::input::InputPlugin::default())
        .add_plugin(bevy::window::WindowPlugin::default())
        .add_plugin(bevy::winit::WinitPlugin::default())
        .add_plugin(bevy_dust::DustPlugin::default())
        .add_plugin(fly_camera::FlyCameraPlugin)
        .add_startup_system(setup_from_oct_file.system())
        .add_system(run.system())
        .run();
}

fn setup_from_oct_file(mut commands: Commands, mut octree: ResMut<Octree>) {
    let file = std::fs::File::open("./test.oct").unwrap();
    let mut reader = std::io::BufReader::new(file);
    Octree::read(&mut octree, &mut reader, 10);

    let mut bundle = RaytracerCameraBundle::default();
    bundle.transform.translation = Vec3::new(1.6, 1.6, 1.6);
    commands
        .spawn()
        .insert_bundle(bundle)
        .insert(FlyCamera::default());
}

fn setup(mut commands: Commands, mut octree: ResMut<Octree>) {
    let mut mutator = octree.get_random_mutator();
    let region_dir = "./assets/region";
    let mut load_region = |region_x: usize, region_y: usize| {
        let file =
            std::fs::File::open(format!("{}/r.{}.{}.mca", region_dir, region_x, region_y)).unwrap();
        let mut region = fastanvil::Region::new(file);

        region
            .for_each_chunk(|chunk_x, chunk_z, chunk_data| {
                println!("loading chunk {} {}", chunk_x, chunk_z);
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
                                    _ => Voxel::with_id(1),
                                };
                                mutator.set(
                                    x + chunk_x as u32 * 16 + region_x as u32 * 512,
                                    y,
                                    z + chunk_z as u32 * 16 + region_y as u32 * 512,
                                    1024,
                                    voxel,
                                );
                            }
                        }
                    }
                }
            })
            .unwrap();
    };

    load_region(1, 0);
    load_region(0, 0);
    load_region(1, 1);
    load_region(0, 1);
    drop(mutator);
    let file = std::fs::File::create("./test.oct").unwrap();
    let mut writer = std::io::BufWriter::new(file);
    octree.write(&mut writer);

    let mut bundle = RaytracerCameraBundle::default();
    bundle.transform.translation = Vec3::new(1.6, 1.6, 1.6);
    commands
        .spawn()
        .insert_bundle(bundle)
        .insert(FlyCamera::default());
}

fn run(mut sunlight: ResMut<SunLight>, time: Res<Time>) {
    let (sin, cos) = (time.seconds_since_startup() * 2.0).sin_cos();
    sunlight.dir = Vec3::new(sin as f32 * 10.0, -3.0, cos as f32 * 10.0).normalize();
}
