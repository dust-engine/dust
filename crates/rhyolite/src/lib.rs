#![feature(generators, generator_trait)]
#![feature(trait_alias)]
#![feature(negative_impls)]
#![feature(const_trait_impl)]
#![feature(get_mut_unchecked)]
#![feature(result_option_inspect)]
#![feature(alloc_layout_extra)]
#![feature(type_alias_impl_trait)]
#![feature(let_chains)]
#![feature(int_roundings)]
#![feature(unsized_locals)]
#![feature(associated_type_bounds)]

pub use bytemuck::offset_of;
pub use cstr::cstr;

pub extern crate ash;
pub extern crate rhyolite_macro as macros;
pub extern crate self as rhyolite;

pub mod accel_struct;
mod allocator;
pub mod commands;
pub mod debug;
pub mod descriptor;
mod device;
mod dho;
pub use dho::*;
pub mod future;
mod instance;
mod physical_device;
mod pipeline;
pub mod queue;
mod resources;
mod sampler;
mod semaphore;
pub mod shader;
mod surface;
pub mod swapchain;
pub mod utils;

pub use device::{Device, HasDevice};
pub use instance::*;
pub use physical_device::*;
pub use pipeline::*;
pub use queue::*;
pub use resources::*;
pub use sampler::Sampler;
pub use semaphore::*;
pub use surface::*;
pub use swapchain::*;

pub use allocator::Allocator;
// TODO: Test two consequtive reads, with different image layouts.
pub use shader::ReflectedShaderModule;
