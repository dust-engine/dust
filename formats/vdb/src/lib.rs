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

mod accessor;
mod bitmask;
mod node;
mod pool;
mod tree;

pub use bitmask::BitMask;
pub use pool::Pool;
pub use tree::Tree;

pub use node::*;
