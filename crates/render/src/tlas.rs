use std::{marker::PhantomData, sync::Arc};

use bevy_app::{Plugin, Update};
use bevy_ecs::{
    prelude::{Component, Entity},
    query::{Changed, Or, With},
    schedule::IntoSystemConfigs,
    system::{Commands, Query, ResMut, Resource},
};
use rhyolite::{
    accel_struct::AccelerationStructure,
    ash::vk,
    future::{GPUCommandFuture, PerFrameState, RenderRes, SharedDeviceStateHostContainer},
    macros::commands,
    HasDevice, ManagedBuffer,
};
use rhyolite_bevy::Allocator;

use crate::{
    blas::{build_blas_system, BLAS},
    Renderable,
};
use bevy_transform::components::GlobalTransform;
use rhyolite::accel_struct::build::TLASBuildInfo;
use rhyolite::debug::DebugObject;

#[derive(Resource)]
pub struct TLASStore<M = Renderable> {
    geometry_flags: vk::GeometryFlagsKHR,
    build_flags: vk::BuildAccelerationStructureFlagsKHR,
    buffer: ManagedBuffer<vk::AccelerationStructureInstanceKHR>,
    requires_rebuild: bool,
    _marker: PhantomData<M>,
}
impl<M> TLASStore<M> {
    pub fn accel_struct(
        &mut self,
    ) -> impl GPUCommandFuture<Output = RenderRes<Arc<AccelerationStructure>>> {
        let buffer = self.buffer.buffer();
        let requires_rebuild = std::mem::replace(&mut self.requires_rebuild, false);

        let allocator = self.buffer.allocator().clone();
        let num_primitives = self.buffer.len() as u32;
        let geometry_flags = self.geometry_flags;
        let build_flags = self.build_flags;
        commands! { move
            let old_tlas: &mut Option<Arc<AccelerationStructure>> = using!();
            if requires_rebuild && let Some(old_tlas) = old_tlas.as_ref() {
                return RenderRes::new(old_tlas.clone());
            }
            let buffer = buffer.await;

            let mut accel_struct = TLASBuildInfo::new(
                allocator,
                num_primitives,
                geometry_flags,
                build_flags,
            ).build_for(buffer).await;
            accel_struct.inner_mut().set_name("a");
            accel_struct.map(|a| {
                let new_tlas = Arc::new(a);
                *old_tlas = Some(new_tlas.clone());
                new_tlas
            })
        }
    }
}
impl<M> HasDevice for TLASStore<M> {
    fn device(&self) -> &std::sync::Arc<rhyolite::Device> {
        self.buffer.device()
    }
}

#[derive(Component)]
pub struct TLASIndex<M> {
    index: u32,
    _marker: PhantomData<M>,
}

fn tlas_system<M: Component>(
    mut commands: Commands,
    mut store: ResMut<TLASStore<M>>,
    mut query: Query<
        (Entity, &BLAS, &GlobalTransform, Option<&mut TLASIndex<M>>),
        (Or<(Changed<BLAS>, Changed<GlobalTransform>)>, With<M>),
    >,
) {
    for (entity, blas, global_transform, index) in query.iter_mut() {
        let Some(blas) = blas.blas.as_ref() else {
            // BLAS isn't ready yet
            continue;
        };
        store.requires_rebuild = true; // Invalidate existing TLAS
        let mut transform = vk::TransformMatrixKHR { matrix: [0.0; 12] };
        transform.matrix.clone_from_slice(
            &global_transform
                .compute_matrix()
                .transpose()
                .to_cols_array()[0..12],
        );

        let instance = rhyolite::ash::vk::AccelerationStructureInstanceKHR {
            transform,
            instance_custom_index_and_mask: vk::Packed24_8::new(0, u8::MAX),
            instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                0,
                vk::GeometryInstanceFlagsKHR::empty().as_raw() as u8,
            ),
            acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                device_handle: blas.device_address(),
            },
        };
        if let Some(index) = index {
            // Index already allocated
            store.buffer.set(index.index as usize, instance);
        } else {
            let index = store.buffer.len();
            store.buffer.push(instance);
            commands.entity(entity).insert(TLASIndex::<M> {
                index: index as u32,
                _marker: Default::default(),
            });
        };
    }
}

pub struct TLASPlugin<M = Renderable>
where
    M: Component,
{
    /// Geometry flags for the TLAS
    geometry_flags: vk::GeometryFlagsKHR,

    /// Build flags for the TLAS
    build_flags: vk::BuildAccelerationStructureFlagsKHR,
    _marker: PhantomData<M>,
}
impl<M: Component> Default for TLASPlugin<M> {
    fn default() -> Self {
        Self {
            geometry_flags: vk::GeometryFlagsKHR::empty(),
            build_flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
            _marker: PhantomData,
        }
    }
}

impl<M: Component> Plugin for TLASPlugin<M> {
    fn build(&self, app: &mut bevy_app::App) {
        let allocator = app.world.resource::<Allocator>().clone();
        app.add_systems(Update, tlas_system::<M>.after(build_blas_system))
            .insert_resource(TLASStore::<M> {
                geometry_flags: self.geometry_flags,
                build_flags: self.build_flags,
                buffer: ManagedBuffer::new(
                    allocator.into_inner(),
                    vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                ),
                requires_rebuild: false,
                _marker: PhantomData,
            });
    }
}
