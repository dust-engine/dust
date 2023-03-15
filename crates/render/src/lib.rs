#![feature(let_chains)]

use bevy_app::Plugin;
mod deferred_task;
mod material;
mod pipeline;
mod sbt;
mod shader;

pub struct RenderPlugin;

impl Plugin for RenderPlugin {
    fn build(&self, _app: &mut bevy_app::App) {}
}
