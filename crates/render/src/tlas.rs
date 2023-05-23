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
    future::{GPUCommandFuture, RenderRes},
    macros::commands,
    HasDevice, ManagedBufferVec,
};
use rhyolite_bevy::{Allocator, RenderSystems};

use crate::{
    blas::{build_blas_system, BLAS},
    sbt::SbtIndex,
    Renderable,
};
use bevy_transform::components::GlobalTransform;
use rhyolite::accel_struct::build::TLASBuildInfo;
use rhyolite::debug::DebugObject;

#[derive(Resource)]
pub struct TLASStore<M = Renderable> {
    geometry_flags: vk::GeometryFlagsKHR,
    build_flags: vk::BuildAccelerationStructureFlagsKHR,
    buffer: ManagedBufferVec<vk::AccelerationStructureInstanceKHR>,
    requires_rebuild: bool,
    _marker: PhantomData<M>,
}
impl<M> TLASStore<M> {
    pub fn accel_struct(
        &mut self,
    ) -> Option<impl GPUCommandFuture<Output = RenderRes<Arc<AccelerationStructure>>>> {
        let Some(buffer) = self.buffer.buffer() else {
            return None;
        };

        let requires_rebuild = std::mem::replace(&mut self.requires_rebuild, false);
        let accel_struct = TLASBuildInfo::new(
            self.buffer.allocator().clone(),
            self.buffer.len() as u32,
            self.geometry_flags,
            self.build_flags,
        );
        let fut = commands! { move
            let old_tlas: &mut Option<Arc<AccelerationStructure>> = using!();
            let buffer = buffer.await;
            if !requires_rebuild && let Some(old_tlas) = old_tlas.as_ref() {
                retain!(buffer);
                return RenderRes::new(old_tlas.clone());
            }

            let mut accel_struct = accel_struct.build_for(buffer).await;
            accel_struct.inner_mut().set_name(&format!("TLAS for {}", std::any::type_name::<M>())).unwrap();
            accel_struct.map(|a| {
                let new_tlas = Arc::new(a);
                *old_tlas = Some(new_tlas.clone());
                new_tlas
            })
        };
        Some(fut)
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
        (
            Entity,
            &BLAS,
            &SbtIndex<M>,
            &GlobalTransform,
            Option<&mut TLASIndex<M>>,
        ),
        (Or<(Changed<BLAS>, Changed<GlobalTransform>)>, With<M>),
    >,
) {
    for (entity, blas, sbt_index, global_transform, index) in query.iter_mut() {
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
                sbt_index.get_index(),
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
    // TODO: implement removal. Create array from TLASIndex -> Entity in TLASStore. Using removal detection,
    // for each removed entity, move last ones to front, use the entity tag to change the TLASIndex stored in ECS.
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
        app.add_systems(
            Update,
            tlas_system::<M>
                .after(build_blas_system)
                .in_set(RenderSystems::SetUp),
        )
        .insert_resource(TLASStore::<M> {
            geometry_flags: self.geometry_flags,
            build_flags: self.build_flags,
            buffer: ManagedBufferVec::new(
                allocator.into_inner(),
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                // VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03715
                // For any element of pInfos[i].pGeometries with a geometryType of VK_GEOMETRY_TYPE_INSTANCES_KHR,
                // if geometry.arrayOfPointers is VK_FALSE, geometry.data->deviceAddress must be aligned to 16 bytes
                16,
            ),
            requires_rebuild: false,
            _marker: PhantomData,
        });
    }
}
