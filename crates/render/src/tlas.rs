use std::{marker::PhantomData, sync::Arc};

use bevy_ecs::{
    prelude::{Component, Entity},
    query::{Changed, With},
    system::{Commands, Query, ResMut, Resource},
};
use rhyolite::{
    accel_struct::AccelerationStructure,
    ash::vk,
    future::{GPUCommandFuture, PerFrameState, RenderRes, SharedDeviceStateHostContainer},
    macros::commands,
    HasDevice, ManagedBuffer,
};

use crate::blas::BLAS;
use bevy_transform::components::GlobalTransform;
use rhyolite::accel_struct::build::TLASBuildInfo;
use rhyolite::future::GPUCommandFutureExt;

#[derive(Resource)]
pub struct TLASStore<M> {
    geometry_flags: vk::GeometryFlagsKHR,
    build_flags: vk::BuildAccelerationStructureFlagsKHR,
    available_indices: Vec<u32>,
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

        let build_info = TLASBuildInfo::new(
            self.buffer.allocator().clone(),
            self.buffer.len() as u32,
            self.geometry_flags,
            self.build_flags,
        );
        commands! { move
            let old_tlas: &mut Option<Arc<AccelerationStructure>> = using!();
            if requires_rebuild && let Some(old_tlas) = old_tlas.as_ref() {
                return RenderRes::new(old_tlas.clone());
            }
            let buffer = buffer.await;
            let accel_struct = build_info.build_for(buffer).await;
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
        (Changed<BLAS>, Changed<GlobalTransform>, With<M>),
    >,
) {
    for (entity, blas, global_transform, index) in query.iter_mut() {
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
                device_handle: blas.blas.device_address(),
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
