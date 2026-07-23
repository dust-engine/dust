#![feature(generic_const_exprs)]
#![feature(adt_const_params)]
#![feature(maybe_uninit_array_assume_init)]
#![feature(const_trait_impl)]
#![feature(generic_const_items)]

mod accessor;
mod attributes;
mod node;

pub mod pool;
mod traversal;
mod tree;

pub use tree::Tree;

pub use accessor::*;
pub use attributes::{AttributeAllocator, Attributes, IsDefault};
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
