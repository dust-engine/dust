use bevy::prelude::*;
use dust_vox::VoxPlugin;

fn main() {
    let mut app = bevy::app::App::new();
    app.add_plugins(bevy::DefaultPlugins.set::<bevy::asset::AssetPlugin>(
        bevy::asset::AssetPlugin {
            mode: bevy::asset::AssetMode::Processed,
            ..Default::default()
        },
    ));

    app.add_plugins(VoxPlugin);

    let scene: Handle<Scene> = app.world.resource::<AssetServer>().load("castle.vox");
    std::mem::forget(scene);
    app.run();
}
