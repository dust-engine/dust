use std::{
    alloc::Layout,
    collections::{BTreeMap, BTreeSet, HashMap},
    sync::Arc, marker::PhantomData,
};

use bevy_app::Plugin;
use bevy_ecs::{prelude::Component, system::Resource};
use rhyolite::{
    ash::vk,
    future::{PerFrameState, SharedDeviceStateHostContainer, GPUCommandFuture, RenderRes},
    Allocator, ResidentBuffer,
    ManagedBufferUnsized, ManagedBufferInner, HasDevice
};

use crate::{Material, Renderable, RayTracingPipelineManagerSpecializedPipeline, RayTracingPipeline, RayTracingPipelineCharacteristics};

// This is to be included on the component of entities.
#[derive(Component)]
pub struct SbtIndex<M = Renderable> {
    index: u32,
    _marker: PhantomData<M>
}


impl<M> Clone for SbtIndex<M> {
    fn clone(&self) -> Self {
        Self {
            index: self.index,
            _marker: PhantomData
        }
    }
}
impl<M> Copy for SbtIndex<M> {
}
impl<M> PartialEq for SbtIndex<M> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}
impl<M> Eq for SbtIndex<M> {}

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

#[derive(Clone, PartialEq, Eq, Hash)]
struct Entry {
    material_id: std::any::TypeId,
    data: Box<[u8]>
}

pub struct SbtManager {
    allocator: Allocator,
    layout: SbtLayout,
    total_raytype: u32,
    buffer: ManagedBufferUnsized,


    /// Mapping from SBT Entry to index
    entries: HashMap<Entry, u32>,
    raytype_pipeline_handles: Vec<vk::Pipeline>,

    update_list: Vec<Entry>,
}

impl SbtManager {
    pub fn new(allocator: rhyolite_bevy::Allocator, pipeline_characteristics: &RayTracingPipelineCharacteristics) -> Self {
        let rtx_properties = allocator.device().physical_device().properties().ray_tracing;
        let handle_layout = Layout::from_size_align(rtx_properties.shader_group_handle_size as usize, rtx_properties.shader_group_handle_alignment as usize).unwrap();
        let one_raytype = handle_layout.extend(pipeline_characteristics.sbt_param_layout).unwrap().0;
        let one_entry = one_raytype.repeat(pipeline_characteristics.num_raytype as usize).unwrap().0;
        let layout = SbtLayout {
            one_raytype,
            one_entry,
            handle_size: rtx_properties.shader_group_handle_size as usize,
        };
        Self {
            allocator: allocator.clone().into_inner(),
            total_raytype: pipeline_characteristics.num_raytype,
            layout,
            buffer: ManagedBufferUnsized::new(allocator.into_inner(), vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR, one_raytype),
            entries: Default::default(),
            raytype_pipeline_handles: vec![vk::Pipeline::null(); pipeline_characteristics.num_raytype as usize],
            update_list: Default::default()
        }
    }
    /// Specify that raytype is using the specialized pipeline for rendering
    pub fn specify_pipelines(&mut self, pipelines: &[RayTracingPipelineManagerSpecializedPipeline]) {
        let mut buffer: Box<[u8]> = vec![0; self.layout.one_raytype.pad_to_align().size()].into_boxed_slice();

        for raytype in 0..self.total_raytype {
            let pipeline = &pipelines[raytype as usize];
            if self.raytype_pipeline_handles[raytype as usize] == pipeline.raw_pipeline() {
                for entry in self.update_list.iter() {
                    let index = self.entries.get(entry).unwrap();
                    let a = pipeline.get_sbt_handle(entry.material_id, raytype);
                    buffer[0..self.layout.handle_size].copy_from_slice(a);
        
                    let size_for_one = entry.data.len() / self.total_raytype as usize;
                    buffer[self.layout.handle_size .. self.layout.handle_size + size_for_one].copy_from_slice(&entry.data[size_for_one * raytype as usize .. size_for_one * (raytype as usize + 1)]);
                    self.buffer.set((*index * self.total_raytype + raytype) as usize, &buffer);
                }
            } else {
                // Update all
                for (entry, index) in self.entries.iter() {
                    let a = pipeline.get_sbt_handle(entry.material_id, raytype);
                    buffer[0..self.layout.handle_size].copy_from_slice(a);
        
                    let size_for_one = entry.data.len() / self.total_raytype as usize;
                    buffer[self.layout.handle_size .. self.layout.handle_size + size_for_one].copy_from_slice(&entry.data[size_for_one * raytype as usize .. size_for_one * (raytype as usize + 1)]);
                    self.buffer.set((*index * self.total_raytype + raytype) as usize, &buffer);
                }
            }
        }
        self.update_list.clear();
    }
    pub fn add_instance<M: Material, A>(&mut self, material: &M) -> SbtIndex<A> {
        let mut data: Box<[u8]> = vec![0; self.total_raytype as usize * std::mem::size_of::<M::ShaderParameters>()].into_boxed_slice();
        for i in 0..self.total_raytype {
            let params = material.parameters(i);

            let size = std::mem::size_of::<M::ShaderParameters>();
            data[size * i as usize .. size * (i as usize + 1)].copy_from_slice(unsafe {
                std::slice::from_raw_parts(&params as *const _ as *const u8, std::mem::size_of_val(&params))
            });
        }
        let entry = Entry {
            material_id: std::any::TypeId::of::<M>(),
            data
        };
        if let Some(existing_index) = self.entries.get(&entry) {
            SbtIndex {
                index: *existing_index,
                _marker: PhantomData
            }
        } else {
            let i = self.buffer.len() as u32;
            self.entries.insert(entry.clone(), i);
            self.update_list.push(entry);
            SbtIndex {
                index: i,
                _marker: PhantomData
            }
        }
    }
}
