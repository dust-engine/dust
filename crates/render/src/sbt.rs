use std::{sync::Arc, alloc::Layout, collections::{BTreeMap, BTreeSet}};

use bevy_ecs::prelude::Component;
use rhyolite::{Allocator, ash::vk, ResidentBuffer, future::{PerFrameState, SharedDeviceStateHostContainer}};

use crate::Material;

struct SbtIndexInner {
    id: usize,
    sender: std::sync::mpsc::Sender<usize>,
}
impl Drop for SbtIndexInner {
    fn drop(&mut self) {
        self.sender.send(self.id).unwrap();
    }
}
unsafe impl Sync for SbtIndexInner {}

// This is to be included on the component of entities.
#[derive(Clone, Component)]
pub struct SbtIndex(Arc<SbtIndexInner>);

struct SbtLayout {
    /// The layout for one raytype.
    /// | Raytype 1                                    |
    /// | shader_handles | inline_parameters | padding |
    /// | <--              size           -> | align   |
    one_raytype: Layout,

    // The layout for one entry with all its raytypes
    /// | Raytype 1                                    | Raytype 2                                    |
    /// | shader_handles | inline_parameters | padding | shader_handles | inline_parameters | padding |
    /// | <---                                      size                               ---> |  align  |
    one_entry: Layout,

    /// The size of the shader group handles, padded.
    /// | Raytype 1                                    |
    /// | shader_handles | inline_parameters | padding |
    /// | <--- size ---> |
    handle_size: usize,
}


pub struct SbtManager {
    allocator: Allocator,
    layout: SbtLayout,
    total_raytype: u32,
    
    sender: std::sync::mpsc::Sender<usize>,
    available_indices: std::sync::mpsc::Receiver<usize>,

    sbt: Vec<u8>,
    sbt_index_to_material: Vec<std::any::TypeId>,


    changeset: BTreeMap<vk::Buffer, BTreeSet<usize>>,
    frames: PerFrameState<ResidentBuffer>,
    /// If None, `frames` are assumed to be device-local and are to be used directly by the GPU.
    device_buffer: Option<SharedDeviceStateHostContainer<ResidentBuffer>>,
}

impl SbtManager {
    pub fn add<M: Material>(&mut self, _hitgroup_id: u32, _parameterse: &[u8]) {
        // for each unique combination of (hitgroup_id, parameters), return unique sbt index.
    }
}
