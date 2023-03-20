#![feature(let_chains)]

use bevy_app::Plugin;
mod deferred_task;
mod material;
mod pipeline;
mod sbt;
mod shader;
mod geometry;

pub struct RenderPlugin;

impl Plugin for RenderPlugin {
    fn build(&self, _app: &mut bevy_app::App) {}
}



pub use material::*;
pub use pipeline::*;
pub use shader::*;