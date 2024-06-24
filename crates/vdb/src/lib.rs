#![feature(generic_const_exprs)]
#![feature(adt_const_params)]
#![feature(maybe_uninit_uninit_array)]
#![feature(maybe_uninit_array_assume_init)]
#![feature(alloc_layout_extra)]

mod accessor;
mod bitmask;
mod node;
mod parry;
mod pool;
mod tree;

pub use bitmask::BitMask;
pub use pool::Pool;
pub use tree::Tree;

pub use accessor::Accessor;
pub use node::*;

pub extern crate self as dust_vdb;

#[derive(Clone, Copy, PartialEq, Eq, std::marker::ConstParamTy)]
pub struct ConstUVec3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl ConstUVec3 {
    pub const fn to_glam(self) -> glam::UVec3 {
        glam::UVec3 {
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }
}
