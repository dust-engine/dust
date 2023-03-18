mod material;
mod palette;
pub use palette::{VoxPalette};
use bevy_asset::Handle;

pub struct PaletteMaterial {
    palette: Handle<VoxPalette>,
    data: Vec<u8>,
}
