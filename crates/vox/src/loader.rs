use std::collections::{BTreeMap, BTreeSet};

use bevy::{
    asset::{AssetLoader, AsyncReadExt},
    math::Vec3A,
    prelude::*,
    utils::{tracing, ConditionalSendFuture},
};
use dot_vox::{DotVoxData, Rotation, SceneNode};
use rayon::prelude::*;
use rhyolite::Allocator;

use crate::{
    attributes::AttributeAllocator, Tree, VoxGeometry, VoxInstance, VoxInstanceBundle, VoxMaterial,
    VoxModelBundle, VoxPalette,
};

enum WorldOrParent<'w, 'q> {
    World(&'w mut World),
    Parent(&'w mut WorldChildBuilder<'q>),
}

impl<'w, 'q> WorldOrParent<'w, 'q> {
    fn spawn(self, bundle: impl Bundle + Send + Sync + 'static) -> EntityWorldMut<'w> {
        match self {
            WorldOrParent::World(world) => world.spawn(bundle),
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

struct SceneGraphTraverser<'a> {
    unit_size: f32,
    scene: &'a DotVoxData,
    models: BTreeSet<u32>,
    instances: Vec<(u32, Entity)>,
}

impl<'a> SceneGraphTraverser<'a> {
    fn traverse(
        &mut self,
        node: u32,
        parent: WorldOrParent<'_, '_>,
        translation: IVec3,
        rotation: Rotation,
        name: Option<&str>,
    ) {
        if self.scene.scenes.is_empty() {
            // Shape nodes are leafs and correspond to models
            assert_eq!(self.scene.models.len(), 1);
            let model = &self.scene.models[0];
            if model.voxels.len() == 0 {
                return;
            }
            let entity = parent
                .spawn(VoxInstanceBundle {
                    transform: Transform::default(),
                    global_transform: GlobalTransform::default(),
                    instance: VoxInstance {
                        model: Entity::from_raw(u32::MAX),
                    },
                })
                .id();
            self.instances.push((0, entity));
            self.models.insert(0);
            return;
        }
        self.traverse_recursive(node, parent, translation, rotation, name);
    }
    fn traverse_recursive(
        &mut self,
        node: u32,
        parent: WorldOrParent<'_, '_>,
        translation: IVec3,
        rotation: Rotation,
        _name: Option<&str>,
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
                let this_translation = frame
                    .position()
                    .map(|position| IVec3 {
                        x: position.x,
                        y: position.y,
                        z: position.z,
                    })
                    .unwrap_or(IVec3::ZERO);

                let this_rotation = frame.orientation().unwrap_or(Rotation::IDENTITY);
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
                        GlobalTransform::default(),
                    ))
                    .with_children(|builder| {
                        for &i in children {
                            self.traverse_recursive(
                                i,
                                WorldOrParent::Parent(builder),
                                IVec3::ZERO,
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
                    .spawn(VoxInstanceBundle {
                        transform: self.to_transform(
                            translation,
                            rotation,
                            UVec3 {
                                x: size.x,
                                y: size.y,
                                z: size.z,
                            },
                        ),
                        ..Default::default()
                    })
                    .id();
                self.instances.push((shape_model.model_id, entity));
                self.models.insert(shape_model.model_id);
            }
        }
    }

    fn to_transform(&self, translation: IVec3, rotation: Rotation, size: UVec3) -> Transform {
        let mut translation = translation.as_vec3a().xzy();
        translation.z *= -1.0;

        let (quat, scale) = rotation.to_quat_scale();
        let quat = Quat::from_array(quat);
        let quat = Quat::from_xyzw(quat.x, quat.z, -quat.y, quat.w);
        let scale = Vec3A::from_array(scale).xzy(); // no need to negate scale.y because scale is not a coordinate

        let mut offset = Vec3A::new(
            if size.x % 2 == 0 { 0.0 } else { 0.5 },
            if size.z % 2 == 0 { 0.0 } else { 0.5 },
            if size.y % 2 == 0 { 0.0 } else { -0.5 },
        );
        offset = quat.mul_vec3a(offset); // If another seam shows up in the future, try multiplying this with `scale`
        let center = quat * (size.xzy().as_vec3a() / 2.0);
        Transform {
            translation: (translation - center * scale + offset).into(),
            rotation: quat,
            scale: scale.into(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum VoxLoadingError {
    #[error("parse error: {0}")]
    ParseError(&'static str),
    #[error("io error: {0}")]
    IoError(#[from] std::io::Error),
}

pub struct VoxLoader {
    allocator: Allocator,
}
impl FromWorld for VoxLoader {
    fn from_world(world: &mut World) -> Self {
        Self {
            allocator: world.resource::<Allocator>().clone(),
        }
    }
}

impl AssetLoader for VoxLoader {
    type Asset = Scene;
    type Settings = ();
    type Error = VoxLoadingError;
    fn load<'a>(
        &'a self,
        reader: &'a mut bevy::asset::io::Reader,
        _settings: &'a Self::Settings,
        load_context: &'a mut bevy::asset::LoadContext,
    ) -> impl ConditionalSendFuture<Output = Result<Scene, VoxLoadingError>> {
        async {
            tracing::info!("Loading vox file {}", load_context.path().display());
            let mut buffer = Vec::new();
            reader.read_to_end(&mut buffer).await?;
            let mut file = dot_vox::load_bytes(buffer.as_slice())
                .map_err(|reason| VoxLoadingError::ParseError(reason))?;
            tracing::info!("Vox file deserialized: {} models", file.models.len());

            let unit_size: f32 = 1.0;

            let mut world = World::default();
            let mut traverser = SceneGraphTraverser {
                unit_size,
                scene: &file,
                models: BTreeSet::new(),
                instances: Vec::new(),
            };
            traverser.traverse(
                0,
                WorldOrParent::World(&mut world),
                IVec3::ZERO,
                Rotation::IDENTITY,
                None,
            );
            let referenced_models = std::mem::take(&mut traverser.models);
            let referenced_instances = std::mem::take(&mut traverser.instances);
            drop(traverser);

            tracing::info!(
                "Scene graph traversed: {} models, {} instances",
                referenced_models.len(),
                referenced_instances.len()
            );

            let palette_handle = load_context.add_labeled_asset(
                "Palette".into(),
                VoxPalette(unsafe {
                    let arr = std::mem::take(&mut file.palette).into_boxed_slice();
                    assert_eq!(arr.len(), 256);
                    Box::from_raw(Box::into_raw(arr) as *mut [_; 256])
                }),
            );

            let model_handles = {
                // Add models
                let mut models: Vec<_> = std::mem::take(&mut file.models)
                    .into_iter()
                    .map(|a| Some(a))
                    .collect();
                let models = referenced_models
                    .iter()
                    .map(|model_id| {
                        (
                            *model_id,
                            models.get_mut(*model_id as usize).unwrap().take().unwrap(),
                        )
                    })
                    .collect::<Vec<_>>();
                let handles = models
                    .par_iter()
                    .map(|(model_id, model)| {
                        let (tree, attribute_allocator) = self.model_to_tree(model);
                        (*model_id, (tree, attribute_allocator))
                    })
                    .collect_vec_list();
                let bundles =
                    handles
                        .into_iter()
                        .flat_map(|a| a)
                        .map(|(model_id, (tree, material))| {
                            let geometry = load_context.add_labeled_asset(
                                format!("Geometry{}", model_id),
                                VoxGeometry { tree, unit_size },
                            );
                            let material = load_context
                                .add_labeled_asset(format!("Material{}", model_id), material);
                            let bundle = VoxModelBundle {
                                geometry,
                                material,
                                palette: palette_handle.clone(),
                                marker: crate::VoxModel,
                            };
                            bundle
                        });
                let entities = world.spawn_batch(bundles);
                BTreeMap::from_iter(referenced_models.into_iter().zip(entities))
            };

            referenced_instances
                .into_iter()
                .for_each(|(model_id, entity_id)| {
                    let model_entity = model_handles.get(&model_id).unwrap();

                    let mut entity = world.entity_mut(entity_id);
                    entity.get_mut::<VoxInstance>().as_mut().unwrap().model = *model_entity;
                });
            let scene = bevy::scene::Scene::new(world);

            tracing::info!("Scene spawned");
            Ok(scene)
        }
    }

    fn extensions(&self) -> &[&str] {
        &["vox"]
    }
}
impl VoxLoader {
    fn model_to_tree(&self, model: &dot_vox::Model) -> (Tree, VoxMaterial) {
        let mut tree = crate::Tree::new_with_gpu_mapped_leaves(self.allocator.clone(), model.voxels.len() as u64 * 128); // TODO: 256 is quite arbitrary here. Consider calculating the upperbound more rigorously
        let mut material = VoxMaterial(
            AttributeAllocator::new_with_capacity(
                self.allocator.clone(),
                64 * model.voxels.len() as u64,
                4,
                64,
            )
            .unwrap(),
        );

        // Create 256x256x256 grid
        let mut accessor = tree.accessor_mut(&mut material);
        let size_y = model.size.y;

        let mut min = UVec3::MAX;
        let mut max = UVec3::MIN;
        for voxel in model.voxels.iter() {
            let voxel = dot_vox::Voxel {
                x: voxel.x,
                y: voxel.z,
                z: (size_y - voxel.y as u32 - 1) as u8,
                i: voxel.i,
            };
            let coords: UVec3 = UVec3 {
                x: voxel.x as u32,
                y: voxel.y as u32,
                z: voxel.z as u32,
            };

            accessor.set(coords, voxel.i + 1);
            min = min.min(coords);
            max = max.max(coords);
        }

        accessor.end();
        material.0.buffer_mut().flush(..);

        (tree, material)
    }
}
