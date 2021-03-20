#![feature(array_methods)]

pub use gfx_hal as hal;

#[cfg(feature = "vulkan")]
pub use gfx_backend_vulkan as back;

#[cfg(feature = "dx12")]
pub use gfx_backend_dx12 as back;


mod renderer;
mod raytracer;
pub use renderer::Renderer;
pub use renderer::Config;