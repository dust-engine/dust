#![feature(generic_const_exprs)]
#![feature(adt_const_params)]
#![feature(maybe_uninit_uninit_array)]
#![feature(maybe_uninit_array_assume_init)]
#![feature(alloc_layout_extra)]
#![feature(mapped_lock_guards)]
#![feature(let_chains)]

mod accessor;
mod bitmask;
mod immutable;
mod node;

#[cfg(feature = "physics")]
mod parry;
mod pool;
mod traversal;
mod tree;

pub use bitmask::BitMask;
pub use pool::Pool;
pub use tree::{MutableTree, TreeLike};

pub use accessor::Accessor;
pub use immutable::*;
pub use node::*;

#[cfg(feature = "physics")]
pub use parry::{VdbQueryDispatcher, VdbShape};

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

#[derive(Clone, Copy, Debug)]
pub struct Aabb<T> {
    pub min: T,
    pub max: T,
}
pub type AabbU16 = Aabb<glam::U16Vec3>;
pub type AabbU32 = Aabb<glam::UVec3>;
impl Default for AabbU16 {
    fn default() -> Self {
        Aabb {
            min: glam::U16Vec3::MAX,
            max: glam::U16Vec3::MIN,
        }
    }
}
impl Default for AabbU32 {
    fn default() -> Self {
        Aabb {
            min: glam::UVec3::MAX,
            max: glam::UVec3::MIN,
        }
    }
}
impl From<AabbU16> for AabbU32 {
    fn from(aabb: AabbU16) -> Self {
        Aabb {
            min: aabb.min.into(),
            max: aabb.max.into(),
        }
    }
}
