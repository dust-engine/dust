use std::sync::Arc;

use bevy::prelude::*;
use bevy_rapier3d::{geometry::Collider, parry::shape::{Shape, SharedShape}};
use dust_vdb::VdbShape;

use crate::{Tree, VoxGeometry, VoxInstance, VoxModel};

pub(crate) fn insert_collider_system(
    mut commands: Commands,
    instances: Query<(Entity, &VoxInstance), Without<Collider>>,
    models: Query<&Handle<VoxGeometry>, With<VoxModel>>,
    geometries: Res<Assets<VoxGeometry>>,
) {
    for (entity, instance) in instances.iter() {
        let Ok(geometry_handle) = models.get(instance.model) else {
            continue;
        };
        let Some(geometry) = geometries.get(geometry_handle) else {
            continue;
        };
        let shape = SharedShape::new(VdbShape::new(Arc::new(geometry.tree.snapshot())));
        commands.entity(entity).insert(Collider::from(shape));
    }
}
