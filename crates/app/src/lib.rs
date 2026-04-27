mod bazel_asset;

use bevy::prelude::*;
use pumicite::{Allocator, ash::vk, swapchain::SwapchainColorMode, tracking::Access};

#[derive(Resource)]
pub struct ApplicationSettings {
    pub project_dirs: directories::ProjectDirs,
}
impl Default for ApplicationSettings {
    fn default() -> Self {
        let project_dirs = directories::ProjectDirs::from("", "DustEngine", "DustBrowser").unwrap();
        // TODO: have more customized behavior for this.
        Self { project_dirs }
    }
}

pub struct DustApp;
impl Plugin for DustApp {
    fn build(&self, app: &mut App) {
        app.init_resource::<ApplicationSettings>();

        app.register_asset_source("bazel", bazel_asset::bazel_asset_source());
    }
}
