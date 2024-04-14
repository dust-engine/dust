use std::{collections::{BTreeMap, BTreeSet}, ops::{Deref, DerefMut}};

use bevy::{
    asset::{Asset, AssetLoader, AsyncReadExt, Handle},
    ecs::{
        bundle::Bundle,
        entity::Entity,
        world::{EntityWorldMut, World},
    },
    hierarchy::{BuildWorldChildren, WorldChildBuilder},
    math::{IVec3, Quat, UVec3, Vec3A, Vec3Swizzles},
    reflect::TypePath,
    transform::components::{GlobalTransform, Transform},
    utils::{tracing, BoxedFuture},
};
use dot_vox::{Color, DotVoxData, Rotation, SceneNode};

use crate::{VoxBundle, VoxModel, VoxPalette};


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
            let entity = parent.spawn(VoxBundle::default()).id();
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

#[derive(Default)]
pub struct VoxLoader;

impl AssetLoader for VoxLoader {
    type Asset = bevy::scene::Scene;
    type Settings = ();
    type Error = VoxLoadingError;
    fn load<'a>(
        &'a self,
        reader: &'a mut bevy::asset::io::Reader,
        _settings: &'a Self::Settings,
        load_context: &'a mut bevy::asset::LoadContext,
    ) -> BoxedFuture<'a, Result<bevy::scene::Scene, VoxLoadingError>> {
        Box::pin(async {
            tracing::info!("Loading vox file {}", load_context.path().display());
            let mut buffer = Vec::new();
            reader.read_to_end(&mut buffer).await?;
            let mut file = dot_vox::load_bytes(buffer.as_slice())
                .map_err(|reason| VoxLoadingError::ParseError(reason))?;
            tracing::info!("Vox file deserialized: {} models", file.models.len());

            let mut world = World::default();
            let mut traverser = SceneGraphTraverser {
                unit_size: 1.0,
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

            let model_handles = {
                // Add models
                let mut models: Vec<_> = std::mem::take(&mut file.models)
                    .into_iter()
                    .map(|a| Some(a))
                    .collect();
                let mut handles: BTreeMap<u32, Handle<VoxModel>> = BTreeMap::default();
                for &model_id in referenced_models.iter() {
                    let Some(model) = models.get_mut(model_id as usize) else {
                        return Err(VoxLoadingError::ParseError("Model not found"));
                    };
                    let model = model.take().unwrap();
                    let handle = load_context
                        .add_labeled_asset(format!("Model{}", model_id), VoxModel(model));
                    handles.insert(model_id, handle);
                }
                handles
            };

            let palette_handle = load_context.add_labeled_asset(
                "Palette".into(),
                VoxPalette(std::mem::take(&mut file.palette)),
            );

            referenced_instances
                .into_iter()
                .for_each(|(model_id, entity_id)| {
                    let handle = model_handles.get(&model_id).unwrap();

                    let mut entity = world.entity_mut(entity_id);
                    *entity.get_mut::<Handle<VoxModel>>().unwrap() = handle.clone();
                    *entity.get_mut::<Handle<VoxPalette>>().unwrap() = palette_handle.clone();
                });
            let scene = bevy::scene::Scene::new(world);

            tracing::info!("Scene spawned");
            Ok(scene)
        })
    }

    fn extensions(&self) -> &[&str] {
        &["vox"]
    }
}
