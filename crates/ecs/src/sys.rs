//! Raw bindings to the flecs C API, covering only what has been called so far.
//! Declarations follow `distr/flecs.h` in the `@flecs` repository.
#![allow(non_camel_case_types, non_upper_case_globals)]

use std::ffi::{c_char, c_int, c_void};

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

pub type ecs_xtor_t =
    unsafe extern "C" fn(ptr: *mut c_void, count: i32, type_info: *const ecs_type_info_t);

pub const ECS_TYPE_HOOK_CTOR_ILLEGAL: u32 = 1 << 10;
pub const ECS_TYPE_HOOK_COPY_ILLEGAL: u32 = 1 << 13;
pub const ECS_TYPE_HOOK_COPY_CTOR_ILLEGAL: u32 = 1 << 15;

/// Only `dtor` is ever set, so the other callbacks are left untyped.
#[repr(C)]
pub struct ecs_type_hooks_t {
    pub ctor: *const c_void,
    pub dtor: Option<ecs_xtor_t>,
    pub copy: *const c_void,
    pub move_: *const c_void,
    pub copy_ctor: *const c_void,
    pub move_ctor: *const c_void,
    pub ctor_move_dtor: *const c_void,
    pub move_dtor: *const c_void,
    pub cmp: *const c_void,
    pub equals: *const c_void,
    pub flags: u32,
    pub on_add: *const c_void,
    pub on_set: *const c_void,
    pub on_remove: *const c_void,
    pub on_replace: *const c_void,
    pub on_validate: *const c_void,
    pub ctx: *mut c_void,
    pub binding_ctx: *mut c_void,
    pub lifecycle_ctx: *mut c_void,
    pub ctx_free: *const c_void,
    pub binding_ctx_free: *const c_void,
    pub lifecycle_ctx_free: *const c_void,
}

#[repr(C)]
pub struct ecs_type_info_t {
    pub size: i32,
    pub alignment: i32,
    pub hooks: ecs_type_hooks_t,
    pub component: ecs_entity_t,
    pub name: *const c_char,
    pub refcount: i32,
}

#[repr(C)]
pub struct ecs_component_desc_t {
    pub _canary: i32,
    pub entity: ecs_entity_t,
    pub type_: ecs_type_info_t,
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
    pub fn ecs_component_init(
        world: *mut ecs_world_t,
        desc: *const ecs_component_desc_t,
    ) -> ecs_entity_t;
    pub fn ecs_emplace_id(
        world: *mut ecs_world_t,
        entity: ecs_entity_t,
        component: ecs_id_t,
        size: usize,
        is_new: *mut bool,
    ) -> *mut c_void;
    pub fn ecs_modified_id(world: *mut ecs_world_t, entity: ecs_entity_t, component: ecs_id_t);
    pub fn ecs_add_id(world: *mut ecs_world_t, entity: ecs_entity_t, component: ecs_id_t);
    pub fn ecs_remove_id(world: *mut ecs_world_t, entity: ecs_entity_t, component: ecs_id_t);
    pub fn ecs_has_id(
        world: *const ecs_world_t,
        entity: ecs_entity_t,
        component: ecs_id_t,
    ) -> bool;
    pub fn ecs_get_id(
        world: *const ecs_world_t,
        entity: ecs_entity_t,
        component: ecs_id_t,
    ) -> *const c_void;
    pub fn ecs_get_mut_id(
        world: *const ecs_world_t,
        entity: ecs_entity_t,
        component: ecs_id_t,
    ) -> *mut c_void;
    pub fn flecs_component_ids_index_get() -> i32;
    pub fn flecs_component_ids_get_alive(world: *const ecs_world_t, index: i32) -> ecs_entity_t;
    pub fn flecs_component_ids_set(world: *mut ecs_world_t, index: i32, component: ecs_entity_t);
}
