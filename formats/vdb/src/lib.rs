#![feature(int_roundings)]
#![feature(stdsimd)]
#![feature(generic_const_exprs)]
#![feature(adt_const_params)]
#![feature(maybe_uninit_uninit_array)]
#![feature(alloc_layout_extra)]
#![feature(generic_associated_types)]
#![feature(const_maybe_uninit_uninit_array)]
#![feature(const_maybe_uninit_write)]
#![feature(const_maybe_uninit_assume_init_read)]
#![feature(const_trait_impl)]
#![feature(const_mut_refs)]
#![feature(const_for)]
#![feature(const_intoiterator_identity)]
#![feature(portable_simd)]

mod accessor;
mod bitmask;
mod geometry;
mod node;
mod pool;
mod tree;

use std::marker::PhantomData;

pub use bitmask::BitMask;
pub use pool::Pool;
pub use tree::Tree;

pub use accessor::Accessor;
pub use geometry::{GPUVdbGeometry, VdbGeometry};
pub use node::*;

#[derive(Default)]
pub struct VdbPlugin<ROOT: Node> {
    _marker: PhantomData<ROOT>,
}
impl<ROOT: NodeConst + Send + Sync> bevy_app::Plugin for VdbPlugin<ROOT>
where
    [(); ROOT::LEVEL as usize + 1]: Sized,
{
    fn build(&self, app: &mut bevy_app::App) {
        app.add_plugin(dust_render::geometry::GeometryPlugin::<VdbGeometry<ROOT>>::default());
    }
}
