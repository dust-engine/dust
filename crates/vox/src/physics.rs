use bevy::prelude::*;
use bevy_rapier3d::geometry::Collider;

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
        let size = geometry.size();
        commands.entity(entity).insert(Collider::cuboid(
            size.x as f32,
            size.y as f32,
            size.z as f32,
        ));
    }
}
