use rhyolite::ash::vk;

use bevy::app::{PluginGroup, Update};
use bevy::ecs::system::Local;
use bevy::ecs::{entity::Entity, query::With};
use bevy::window::PrimaryWindow;
use rhyolite::debug::DebugUtilsPlugin;
use rhyolite::{RhyolitePlugin, SurfacePlugin, SwapchainConfig, SwapchainPlugin};
use rhyolite_egui::{egui, EguiContexts};

fn main() {
    let mut app = bevy::app::App::new();
    app.add_plugins(bevy::DefaultPlugins.set::<bevy::asset::AssetPlugin>(
        bevy::asset::AssetPlugin {
            mode: bevy::asset::AssetMode::Processed,
            ..Default::default()
        },
    ))
    .add_plugins(SurfacePlugin::default())
    .add_plugins(DebugUtilsPlugin::default())
    .add_plugins(RhyolitePlugin::default())
    .add_plugins(SwapchainPlugin::default())
    .add_plugins(rhyolite_egui::EguiPlugin::<With<PrimaryWindow>>::default());

    app.add_systems(Update, side_panel_system);

    let primary_window = app
        .world
        .query_filtered::<Entity, With<PrimaryWindow>>()
        .iter(&app.world)
        .next()
        .unwrap();
    app.world
        .entity_mut(primary_window)
        .insert(SwapchainConfig {
            image_usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            ..Default::default()
        });

    app.run();
}

#[derive(Default)]
struct UIState {
    name: String,
    age: u32,

    file_path: Option<String>,
}
fn side_panel_system(mut contexts: EguiContexts, mut state: Local<UIState>) {
    egui::SidePanel::left("left_panel").show(contexts.ctx_mut(), |ui| {
        if ui.button("Open gltf file…").clicked() {
            if let Some(path) = rfd::FileDialog::new().pick_file() {
                state.file_path = Some(path.display().to_string());
            }
        }
        ui.label("Hello World!");
    });
}
