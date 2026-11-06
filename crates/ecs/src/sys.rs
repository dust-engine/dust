//! Raw bindings to the flecs C API, covering only what has been called so far.
//! Declarations follow `distr/flecs.h` in the `@flecs` repository.
#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_int};

pub type ecs_id_t = u64;
pub type ecs_entity_t = ecs_id_t;

#[repr(C)]
pub struct ecs_world_t(());

#[repr(C)]
pub struct ecs_build_info_t {
    pub compiler: *const c_char,
    pub addons: *const *const c_char,
    pub flags: *const *const c_char,
    pub version: *const c_char,
    pub version_major: i16,
    pub version_minor: i16,
    pub version_patch: i16,
    pub debug: bool,
    pub sanitize: bool,
    pub perf_trace: bool,
}

unsafe extern "C" {
    pub fn ecs_get_build_info() -> *const ecs_build_info_t;
    pub fn ecs_init() -> *mut ecs_world_t;
    pub fn ecs_fini(world: *mut ecs_world_t) -> c_int;
    pub fn ecs_new(world: *mut ecs_world_t) -> ecs_entity_t;
    pub fn ecs_delete(world: *mut ecs_world_t, entity: ecs_entity_t);
    pub fn ecs_is_alive(world: *const ecs_world_t, entity: ecs_entity_t) -> bool;
    pub fn ecs_set_name(
        world: *mut ecs_world_t,
        entity: ecs_entity_t,
        name: *const c_char,
    ) -> ecs_entity_t;
    pub fn ecs_get_name(world: *const ecs_world_t, entity: ecs_entity_t) -> *const c_char;
}
