use std::{collections::HashSet, future::Future};

use bevy_ecs::{
    prelude::{Bundle, Entity},
    world::{EntityMut, World, FromWorld},
};
use bevy_hierarchy::{BuildWorldChildren, WorldChildBuilder};
use bevy_transform::prelude::{GlobalTransform, Transform};
use dot_vox::{Color, DotVoxData, Model, SceneNode, Rotation};
use glam::{IVec3, UVec3, Vec3Swizzles};
use rhyolite::{ash::vk, future::{GPUCommandFutureExt, GPUCommandFuture}, QueueRef, macros::commands, commands};
use rayon::prelude::*;
/// MagicaVoxel trees are 256x256x256 max, so the numbers in the
/// hierarchy must sum up to 8 where 2^8 = 256.
use crate::{TreeRoot, Tree, VoxBundle};
use crate::{palette::VoxPalette, VoxGeometry};

use bevy_asset::{AssetLoader, Assets, Handle, LoadedAsset};
use rhyolite_bevy::{AsyncQueues, QueuesRouter};

use crate::material::PaletteMaterial;

pub struct VoxLoader {
    allocator: rhyolite_bevy::Allocator,
    queues: AsyncQueues,
    transfer_queue: QueueRef,
}
impl FromWorld for VoxLoader {
    fn from_world(world: &mut World) -> Self {
        let allocator = world.resource::<rhyolite_bevy::Allocator>().clone();
        let queues = world.resource::<AsyncQueues>().clone();
        let transfer_queue = world.resource::<QueuesRouter>().of_type(rhyolite::QueueType::Transfer);
        Self {
            allocator,
            queues,
            transfer_queue
        }
    }
}

struct SceneGraphTraverser<'a> {
    unit_size: f32,
    scene: &'a DotVoxData,
    models: HashSet<u32>,
    instances: Vec<(u32, Entity)>,
} 

impl<'a> SceneGraphTraverser<'a> {
    fn traverse_recursive(
        &mut self,
        node: u32,
        parent: WorldOrParent<'_, '_>,
        translation: glam::IVec3,
        rotation: Rotation,
        name: Option<&str>,
    ) {
        let node = &self.scene.scenes[node as usize];
        match node {
            SceneNode::Transform {
                attributes,
                frames,
                child,
                layer_id: _,
            } => {
                if frames.len() != 1 {
                    unimplemented!("Multiple frame in transform node");
                }
                let name = attributes.get("_name").map(String::as_str);
                let frame = &frames[0];
                let mut this_translation =
                    frame.position().map(|position| IVec3 { x: position.x, y: position.y, z: position.z }).unwrap_or(IVec3::ZERO);

                let this_rotation = frame
                    .orientation()
                    .unwrap_or(Rotation::IDENTITY);
                //let rotation = rotation * this_rotation; // reverse?
                let translation = translation + this_translation;

                self.traverse_recursive(*child, parent, translation, this_rotation, name);
            }
            SceneNode::Group {
                attributes: _,
                children,
            } => {
                parent
                    .spawn((
                        self.to_transform(translation, rotation, UVec3::ZERO),
                        GlobalTransform::default()
                    ))
                    .with_children(|builder| {
                        for &i in children {
                            self.traverse_recursive(
                                i,
                                WorldOrParent::Parent(builder),
                                glam::IVec3::ZERO,
                                Rotation::IDENTITY,
                                None,
                            );
                        }
                    });
            }
            SceneNode::Shape {
                attributes: _,
                models,
            } => {
                // Shape nodes are leafs and correspond to models
                if models.len() != 1 {
                    unimplemented!("Multiple shape models in Shape node");
                }
                let shape_model = &models[0];
                let model = &self.scene.models[shape_model.model_id as usize];
                if model.voxels.len() == 0 {
                    return;
                }
                let size = self.scene.models[shape_model.model_id as usize].size;
                let entity = parent
                    .spawn(VoxBundle {
                        transform: self.to_transform(
                            translation,
                            rotation,
                            UVec3 {
                                x: size.x,
                                y: size.y,
                                z: size.z,
                            },
                        ),
                        ..VoxBundle::from_geometry_material(Handle::default(), Handle::default())
                    })
                    .id();
                self.instances.push((shape_model.model_id, entity));
                self.models.insert(shape_model.model_id);
            }
        }
    }

    fn to_transform(
        &self,
        translation: glam::IVec3,
        rotation: Rotation,
        size: glam::UVec3,
    ) -> Transform {
        let mut translation = translation.as_vec3a().xzy();
        translation.z *= -1.0;

        let (quat, scale) = rotation.to_quat_scale();
        let quat = glam::Quat::from_array(quat);
        let quat = glam::Quat::from_xyzw(quat.x, quat.z, -quat.y, quat.w);
        let mut scale = glam::Vec3A::from_array(scale).xzy(); // no need to negate scale.y because scale is not a coordinate

        let center = quat * (size.xzy().as_vec3a() / 2.0);
        Transform {
            translation: (translation - center * scale).into(),
            rotation: quat,
            scale: scale.into(),
        }
    }
}

impl VoxLoader {
    fn load_palette(&self, palette: &[dot_vox::Color]) -> VoxPalette {
        unsafe {
            const LEN: usize = 255;
            let mem =
                std::alloc::alloc(std::alloc::Layout::new::<[Color; LEN]>()) as *mut [Color; LEN];
            let mut mem = Box::from_raw(mem);
            mem.copy_from_slice(&palette[0..LEN]);
            VoxPalette(mem)
        }
    }

    fn load_model(&self, model: &Model, palette: Handle<VoxPalette>) -> impl GPUCommandFuture<Output = (VoxGeometry, PaletteMaterial)> +Send{
        let mut palette_index_collector = crate::collector::ModelIndexCollector::new();

        let mut tree = Tree::new();
        for voxel in model.voxels.iter() {
            let voxel = dot_vox::Voxel {
                x: voxel.x,
                y: voxel.z,
                z: model.size.y as u8 - voxel.y,
                i: voxel.i,
            };
            let coords: UVec3 = UVec3 {
                x: voxel.x as u32,
                y: voxel.y as u32,
                z: voxel.z as u32,
            };
            tree.set_value(coords, Some(true));
            palette_index_collector.set(voxel);
        }

        let palette_indexes = palette_index_collector.into_iter();
        // TODO: use iter_leaf_mut here, and insert indices
        for (location, leaf) in tree.iter_leaf_mut() {
            let block_index = (location.x >> 2, location.y >> 2, location.z >> 2);
            let block_index = block_index.0 as usize
                + block_index.1 as usize * 64
                + block_index.2 as usize * 64 * 64;

            leaf.material_ptr = palette_indexes.running_sum()[block_index];
        }

        let material_buffer = self.allocator.create_dynamic_asset_buffer_with_writer(
            palette_indexes.len() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            |slice| {
                for (src, dst) in palette_indexes.zip(slice.iter_mut()) {
                    *dst = src;
                }
            }
        ).unwrap();

        
        let geometry = VoxGeometry::from_tree(
            tree,
            [model.size.x as u8, model.size.z as u8, model.size.y as u8],
            1.0,
            &self.allocator
        );

        let future_to_wait = material_buffer.join(geometry);
        future_to_wait.map(|(buffer, geometry)| {
            let buffer = buffer.into_inner();
            (geometry, PaletteMaterial::new(palette, buffer))
        })
    } 
}

impl AssetLoader for VoxLoader {
    fn load<'a>(
        &'a self,
        bytes: &'a [u8],
        load_context: &'a mut bevy_asset::LoadContext,
    ) -> bevy_asset::BoxedFuture<'a, Result<(), anyhow::Error>> {
        Box::pin(async {
            let file = dot_vox::load_bytes(bytes).map_err(|str| anyhow::Error::msg(str))?;

            let palette = self.load_palette(&file.palette);

            let palette_handle =
                load_context.set_labeled_asset("palette", LoadedAsset::new(palette));

            let mut world = World::default();
            let mut traverser = SceneGraphTraverser {
                unit_size: 1.0,
                scene: &file,
                models: HashSet::new(),
                instances: Vec::new(),
            };
            traverser.traverse_recursive(
                0,
                WorldOrParent::World(&mut world),
                IVec3::ZERO,
                Rotation::IDENTITY,
                None,
            );

            let geometry_material_futures: Vec<_> = traverser.models.par_iter().map(|model_id| {
                let model = &file.models[*model_id as usize];
                assert!(model.size.x <= 255 && model.size.y <= 255 && model.size.z <= 255);

                (*model_id,
                    self.load_model(model, palette_handle.clone()))
            }).collect();
            let geometry_materials = commands! {
                let mut geometry_materials: Vec<_> = Vec::with_capacity(traverser.models.len());
                for (model_id, future) in geometry_material_futures.into_iter() {
                    let (geometry, material) = future.await;
                    geometry_materials.push((model_id, geometry, material));
                }
                geometry_materials
            }.schedule_on_queue(self.transfer_queue);
            let geometry_materials = self.queues.submit(geometry_materials, &mut Default::default()).await;

            let mut models: Vec<Option<(Handle<VoxGeometry>, Handle<PaletteMaterial>)>> =
                vec![None; file.models.len()];
            for (model_id, geometry, material) in geometry_materials.into_iter() {
                let geometry_handle = load_context.set_labeled_asset(
                    &format!("Geometry{}", model_id),
                    LoadedAsset::new(geometry),
                );
                let material_handle = load_context.set_labeled_asset(
                    &format!("Material{}", model_id),
                    LoadedAsset::new(material),
                );
                models[model_id as usize] = Some((geometry_handle, material_handle));
            }
            for (model_id, entity) in traverser.instances.iter() {
                let mut entity = world.entity_mut(*entity);
                let (geometry_handle, material_handle) =
                    models[*model_id as usize].as_ref().unwrap();
                *entity.get_mut::<Handle<VoxGeometry>>().unwrap() = geometry_handle.clone();
                *entity.get_mut::<Handle<PaletteMaterial>>().unwrap() = material_handle.clone();
            }

            let scene = bevy_scene::Scene::new(world);
            load_context.set_default_asset(LoadedAsset::new(scene));
            Ok(())
        })
    }

    fn extensions(&self) -> &[&str] {
        &["vox"]
    }
}

enum WorldOrParent<'w, 'q> {
    World(&'w mut World),
    Parent(&'w mut WorldChildBuilder<'q>),
}

impl<'w, 'q> WorldOrParent<'w, 'q> {
    fn spawn(self, bundle: impl Bundle + Send + Sync + 'static) -> EntityMut<'w> {
        match self {
            WorldOrParent::World(world) => {
                world.spawn(bundle)
            }
            WorldOrParent::Parent(parent) => parent.spawn(bundle),
        }
    }
    fn has_parent(&self) -> bool {
        match self {
            WorldOrParent::World(_) => false,
            WorldOrParent::Parent(_) => true,
        }
    }
}
