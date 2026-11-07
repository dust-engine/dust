//! Raw bindings to the flecs C API, covering only what has been called so far.
//! Declarations follow `distr/flecs.h` in the `@flecs` repository.
#![allow(non_camel_case_types, non_upper_case_globals)]

use std::ffi::{c_char, c_int, c_void};
use std::mem::offset_of;

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

/// The most terms a query can have.
pub const FLECS_TERM_COUNT_MAX: usize = 32;

// `ecs_inout_kind_t`, which `ecs_term_t` stores as an `i16`.
pub const EcsInOutNone: i16 = 1;
pub const EcsInOut: i16 = 3;
pub const EcsIn: i16 = 4;

// `ecs_oper_kind_t`, which `ecs_term_t` stores as an `i16`.
pub const EcsAnd: i16 = 0;
pub const EcsNot: i16 = 2;
pub const EcsOptional: i16 = 3;

pub type ecs_query_cache_kind_t = c_int;
pub const EcsQueryCacheAuto: ecs_query_cache_kind_t = 1;

#[repr(C)]
pub struct ecs_term_ref_t {
    pub id: ecs_entity_t,
    pub name: *const c_char,
}

#[repr(C)]
pub struct ecs_term_t {
    pub id: ecs_id_t,
    pub src: ecs_term_ref_t,
    pub first: ecs_term_ref_t,
    pub second: ecs_term_ref_t,
    pub trav: ecs_entity_t,
    pub inout: i16,
    pub oper: i16,
    pub field_index: i8,
    pub flags_: u16,
}

/// Only terms and the cache kind are ever set, so the callbacks are left untyped.
#[repr(C)]
pub struct ecs_query_desc_t {
    pub _canary: i32,
    pub terms: [ecs_term_t; FLECS_TERM_COUNT_MAX],
    pub expr: *const c_char,
    pub cache_kind: ecs_query_cache_kind_t,
    pub flags: u32,
    pub order_by_callback: *const c_void,
    pub order_by_table_callback: *const c_void,
    pub order_by: ecs_entity_t,
    pub group_by: ecs_id_t,
    pub group_by_callback: *const c_void,
    pub on_group_create: *const c_void,
    pub on_group_delete: *const c_void,
    pub group_by_ctx: *mut c_void,
    pub group_by_ctx_free: *const c_void,
    pub ctx: *mut c_void,
    pub binding_ctx: *mut c_void,
    pub ctx_free: *const c_void,
    pub binding_ctx_free: *const c_void,
    pub entity: ecs_entity_t,
}

#[repr(C)]
pub struct ecs_header_t {
    pub type_: i32,
    pub refcount: i32,
    pub mixins: *mut c_void,
}

/// One bit per field of a query.
pub type ecs_termset_t = u32;

#[repr(C)]
pub struct ecs_query_t {
    pub hdr: ecs_header_t,
    pub terms: *mut ecs_term_t,
    pub sizes: *mut i32,
    pub ids: *mut ecs_id_t,
    pub bloom_filter: u64,
    pub flags: u32,
    pub var_count: i8,
    pub term_count: i8,
    pub field_count: i8,
    pub fixed_fields: ecs_termset_t,
    pub var_fields: ecs_termset_t,
    pub static_id_fields: ecs_termset_t,
    pub data_fields: ecs_termset_t,
    pub write_fields: ecs_termset_t,
    pub read_fields: ecs_termset_t,
    pub row_fields: ecs_termset_t,
    pub shared_readonly_fields: ecs_termset_t,
    pub set_fields: ecs_termset_t,
    pub cache_kind: ecs_query_cache_kind_t,
    pub vars: *mut *mut c_char,
    pub ctx: *mut c_void,
    pub binding_ctx: *mut c_void,
    pub entity: ecs_entity_t,
    pub stage: *mut ecs_world_t,
    pub world: *mut ecs_world_t,
    pub eval_count: i32,
}

#[repr(C)]
pub struct ecs_table_t(());

/// State that only flecs reads.
#[repr(C)]
pub struct ecs_iter_private_t([u64; 21]);

/// Returned by value and only read here, as flecs does the iterating, so its callbacks are
/// left untyped.
#[repr(C)]
pub struct ecs_iter_t {
    pub stage: *mut ecs_world_t,
    pub world: *mut ecs_world_t,
    pub offset: i32,
    pub count: i32,
    pub entities: *const ecs_entity_t,
    pub ptrs: *mut *mut c_void,
    pub trs: *const *const c_void,
    pub columns: *const i16,
    pub sizes: *const i32,
    pub table: *mut ecs_table_t,
    pub other_table: *mut ecs_table_t,
    pub ids: *mut ecs_id_t,
    pub sources: *mut ecs_entity_t,
    pub constrained_vars: u64,
    pub set_fields: ecs_termset_t,
    pub ref_fields: ecs_termset_t,
    pub row_fields: ecs_termset_t,
    pub up_fields: ecs_termset_t,
    pub system: ecs_entity_t,
    pub event: ecs_entity_t,
    pub event_id: ecs_id_t,
    pub event_cur: i32,
    pub field_count: i8,
    pub term_index: i8,
    pub query: *const ecs_query_t,
    pub param: *mut c_void,
    pub ctx: *mut c_void,
    pub binding_ctx: *mut c_void,
    pub callback_ctx: *mut c_void,
    pub run_ctx: *mut c_void,
    pub delta_time: f32,
    pub delta_system_time: f32,
    pub frame_offset: i32,
    pub flags: u32,
    pub interrupted_by: ecs_entity_t,
    pub priv_: ecs_iter_private_t,
    pub next: *const c_void,
    pub callback: *const c_void,
    pub fini: *const c_void,
    pub chain_it: *mut ecs_iter_t,
}

/// In `ecs_iter_t::flags`, set if a query matches through component inheritance.
pub const EcsIterComponentInheritance: u32 = 1 << 10;

/// In `ecs_query_t::flags`, set if a query has terms on `$this`, the entity of each row.
pub const EcsQueryMatchThis: u32 = 1 << 11;

impl ecs_iter_t {
    /// How many rows to visit in the current result. A result with no rows and no table is still
    /// worth one visit, as it's what a query gets when all its values come from fixed sources,
    /// such as singletons.
    pub fn num_rows(&self) -> usize {
        if self.count == 0 && self.table.is_null() {
            1
        } else {
            self.count as usize
        }
    }
}

// As measured in C, where they don't change with `NDEBUG`. A layout mistake links fine and
// then corrupts memory. Measure these again when upgrading flecs, as this only catches a change
// to the Rust side.
const _: () = {
    assert!(size_of::<ecs_term_t>() == 72);
    assert!(size_of::<ecs_query_desc_t>() == 2440);
    assert!(size_of::<ecs_query_t>() == 152);
    assert!(size_of::<ecs_iter_private_t>() == 168);
    assert!(size_of::<ecs_iter_t>() == 424);
    assert!(offset_of!(ecs_query_t, term_count) == 53);
    assert!(offset_of!(ecs_query_t, field_count) == 54);
    assert!(offset_of!(ecs_query_t, stage) == 128);
    assert!(offset_of!(ecs_query_t, world) == 136);
    assert!(offset_of!(ecs_iter_t, stage) == 0);
    assert!(offset_of!(ecs_iter_t, world) == 8);
};

unsafe extern "C" {
    pub safe static FLECS_IDEcsComponentID_: ecs_entity_t;
    pub safe static EcsCanToggle: ecs_entity_t;
    pub safe static EcsSparse: ecs_entity_t;
    pub safe static EcsDontFragment: ecs_entity_t;
    pub safe static EcsSingleton: ecs_entity_t;
    pub safe static EcsWith: ecs_entity_t;

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
    pub fn ecs_make_pair(relationship: ecs_entity_t, target: ecs_entity_t) -> ecs_id_t;
    pub fn ecs_enable(world: *mut ecs_world_t, entity: ecs_entity_t, enabled: bool);
    pub fn ecs_enable_id(
        world: *mut ecs_world_t,
        entity: ecs_entity_t,
        component: ecs_id_t,
        enable: bool,
    );
    pub fn ecs_is_enabled_id(
        world: *const ecs_world_t,
        entity: ecs_entity_t,
        component: ecs_id_t,
    ) -> bool;
    pub fn ecs_add_id(world: *mut ecs_world_t, entity: ecs_entity_t, component: ecs_id_t);
    pub fn ecs_remove_id(world: *mut ecs_world_t, entity: ecs_entity_t, component: ecs_id_t);
    pub fn ecs_has_id(world: *const ecs_world_t, entity: ecs_entity_t, component: ecs_id_t)
    -> bool;
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
    pub fn ecs_query_init(
        world: *mut ecs_world_t,
        desc: *const ecs_query_desc_t,
    ) -> *mut ecs_query_t;
    pub fn ecs_query_fini(query: *mut ecs_query_t);
    pub fn ecs_query_get(world: *const ecs_world_t, query: ecs_entity_t) -> *const ecs_query_t;
    pub fn ecs_query_iter(world: *const ecs_world_t, query: *const ecs_query_t) -> ecs_iter_t;
    pub fn ecs_query_next(it: *mut ecs_iter_t) -> bool;
    pub fn ecs_query_has(
        query: *const ecs_query_t,
        entity: ecs_entity_t,
        it: *mut ecs_iter_t,
    ) -> bool;
    pub fn ecs_query_is_true(query: *const ecs_query_t) -> bool;
    pub fn ecs_iter_fini(it: *mut ecs_iter_t);
    pub fn ecs_field_w_size(it: *const ecs_iter_t, size: usize, index: i8) -> *mut c_void;
    pub fn ecs_base_field_w_size(it: *const ecs_iter_t, size: usize, index: i8) -> *mut c_void;
    pub fn ecs_field_stride(it: *const ecs_iter_t, index: i8) -> usize;
    pub fn ecs_field_at_w_size(
        it: *const ecs_iter_t,
        size: usize,
        index: i8,
        row: i32,
    ) -> *mut c_void;
    pub fn ecs_field_is_set(it: *const ecs_iter_t, index: i8) -> bool;
    pub fn flecs_component_ids_index_get() -> i32;
    pub fn flecs_component_ids_get_alive(world: *const ecs_world_t, index: i32) -> ecs_entity_t;
    pub fn flecs_component_ids_set(world: *mut ecs_world_t, index: i32, component: ecs_entity_t);
}
