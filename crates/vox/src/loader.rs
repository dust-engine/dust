use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use bevy::ecs::name::Name;
use bevy::ecs::template::{EntityTemplate, SceneEntityReference};
use bevy::scene::{NameEntityReference, RelatedScenes, Scene, ScenePatch, bsn, template_value};
use bevy::{
    asset::AssetLoader,
    math::{U8Vec4, Vec3A},
    prelude::*,
};
use dot_vox::{DotVoxData, Rotation, SceneNode};
use pumicite::{Allocator, ash::vk};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{VoxGeometry, VoxGpuMaterial, VoxInstanceTemplate, VoxMaterial, VoxModel, VoxPalette};

pub enum GraphScene {
    Node {
        transform: Transform,
        children: Vec<GraphScene>,
    },
    Leaf {
        transform: Transform,
        model: SceneEntityReference,
    },
}
impl Scene for GraphScene {
    fn resolve(
        self,
        context: &mut bevy::scene::ResolveContext,
        scene: &mut bevy::scene::ResolvedScene,
    ) -> Result<(), bevy::scene::ResolveSceneError> {
        match self {
            Self::Node {
                transform,
                children,
            } => {
                scene.push_template(transform);
                RelatedScenes::<ChildOf, _>::new(children).resolve(context, scene);
                Ok(())
            }
            Self::Leaf { transform, model } => {
                scene.push_template(transform);
                scene.push_template(VoxInstanceTemplate {
                    model: EntityTemplate::SceneEntityReference(model),
                });
                Ok(())
            }
        }
    }
}

impl VoxGpuMaterial {
    /// Decode a MagicaVoxel `MATL` entry into the packed GPU representation.
    ///
    /// Reads the raw `properties` dictionary directly (rather than dot_vox's
    /// typed accessors) so this stays correct regardless of dot_vox version and
    /// sidesteps accessor key quirks (e.g. its `specular()` reads `_sp` while
    /// MagicaVoxel writes `_spec`).
    fn apply_dot_vox(&mut self, material: &dot_vox::Material) {
        let get = |key: &str| -> Option<f32> {
            material
                .properties
                .get(key)
                .and_then(|s| s.parse::<f32>().ok())
        };

        // MagicaVoxel material types: _diffuse (default), _metal, _glass,
        // _emit, _media/_cloud.
        let ty: u32 = match material.properties.get("_type").map(String::as_str) {
            Some("_metal") => 1,
            Some("_glass") => 2,
            Some("_emit") => 3,
            Some("_media") | Some("_cloud") => 4,
            _ => 0,
        };

        let unorm = |x: f32| (x.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
        let roughness = unorm(get("_rough").unwrap_or(0.5));
        // MagicaVoxel encodes a metal's metalness via `_weight`; fall back to it.
        let metalness = unorm(
            get("_metal")
                .or_else(|| (ty == 1).then(|| get("_weight").unwrap_or(1.0)))
                .unwrap_or(0.0),
        );
        let specular = unorm(get("_spec").unwrap_or(0.0));
        self.packed = ty | (roughness << 8) | (metalness << 16) | (specular << 24);

        // Emissive radiance multiplier: base emission scaled by the radiant-flux
        // exponent. The exact curve is a modeling choice; this is a reasonable
        // starting point and is only consumed once emissive lighting lands.
        let emission = if ty == 3 {
            get("_emit").unwrap_or(0.0) * 10f32.powf(get("_flux").unwrap_or(0.0))
        } else {
            0.0
        };
        self.emission = emission as f16;

        // MagicaVoxel's `_ior` is stored relative to vacuum (real IOR ≈ 1 + _ior).
        self.ior = (1.0 + get("_ior").unwrap_or(0.0)) as f16;
        self.transparency = get("_trans").or_else(|| get("_alpha")).unwrap_or(0.0) as f16;
    }
}

struct SceneGraphTraverser<'a> {
    unit_size: f32,
    scene: &'a DotVoxData,
    /// One entity reference per referenced model, allocated on first use.
    models: BTreeMap<u32, SceneEntityReference>,
}

impl<'a> SceneGraphTraverser<'a> {
    /// The entity reference for `model_id`, allocating one on first use.
    fn model_reference(&mut self, model_id: u32) -> SceneEntityReference {
        static VOX_SCENE_COUNTER: AtomicU64 = AtomicU64::new(0);
        let run_count = VOX_SCENE_COUNTER.fetch_add(1, Ordering::Relaxed);
        *self.models.entry(model_id).or_insert_with(|| {
            SceneEntityReference::new(
                (file!(), line!() as usize, column!() as usize),
                model_id as usize,
                run_count,
            )
        })
    }

    fn traverse(
        &mut self,
        node: u32,
        out: &mut Vec<GraphScene>,
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
            let reference = self.model_reference(0);
            out.push(GraphScene::Leaf {
                transform: Transform::default(),
                model: reference,
            });
            return;
        }
        self.traverse_recursive(node, out, translation, rotation, name);
    }
    fn traverse_recursive(
        &mut self,
        node: u32,
        out: &mut Vec<GraphScene>,
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

                self.traverse_recursive(*child, out, translation, this_rotation, name);
            }
            SceneNode::Group {
                attributes: _,
                children,
            } => {
                let transform = self.to_transform(translation, rotation, UVec3::ZERO);
                let mut group_children = Vec::new();
                for &i in children {
                    self.traverse_recursive(
                        i,
                        &mut group_children,
                        IVec3::ZERO,
                        Rotation::IDENTITY,
                        None,
                    );
                }
                out.push(GraphScene::Node {
                    transform,
                    children: group_children,
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
                let transform = self.to_transform(
                    translation,
                    rotation,
                    UVec3 {
                        x: size.x,
                        y: size.y,
                        z: size.z,
                    },
                );
                let reference = self.model_reference(shape_model.model_id);
                out.push(GraphScene::Leaf {
                    transform,
                    model: reference,
                });
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
        // BLAS AABBs are in object-space units of `unit_size` per voxel cell, but
        // `translation`, `center`, and `offset` are computed in voxel-grid units
        // from the MagicaVoxel scene graph. Scale them so the world-space placement
        // matches the world-space size of the model.
        let unit_size = self.unit_size;
        Transform {
            translation: ((translation - center * scale + offset) * unit_size).into(),
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
    #[error("vulkan error: {0}")]
    VulkanError(#[from] vk::Result),
}

#[derive(TypePath)]
pub struct VoxLoader {
    allocator: Allocator,
}
impl VoxLoader {
    pub fn new(allocator: Allocator) -> Self {
        Self { allocator }
    }
}
impl FromWorld for VoxLoader {
    fn from_world(world: &mut World) -> Self {
        Self {
            allocator: world.resource::<Allocator>().clone(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VoxLoaderSettings {
    pub unit_size: f32,
}
impl Default for VoxLoaderSettings {
    fn default() -> Self {
        // The default scale was set up this way because 1 minecraft block has 16x16 texture.
        // so if unit_size is 1.0 / 16.0, we have one voxel for every pixel on a minecraft block.
        Self {
            unit_size: 1.0 / 16.0,
        }
    }
}

impl AssetLoader for VoxLoader {
    type Asset = ScenePatch;
    type Settings = VoxLoaderSettings;
    type Error = VoxLoadingError;
    fn load(
        &self,
        reader: &mut dyn bevy::asset::io::Reader,
        settings: &Self::Settings,
        load_context: &mut bevy::asset::LoadContext,
    ) -> impl bevy::tasks::ConditionalSendFuture<Output = Result<ScenePatch, VoxLoadingError>> {
        async {
            tracing::info!("Loading vox file {}", load_context.path());
            let mut buffer = Vec::new();
            reader.read_to_end(&mut buffer).await?;
            let mut file = dot_vox::load_bytes(buffer.as_slice())
                .map_err(|reason| VoxLoadingError::ParseError(reason))?;
            tracing::info!("Vox file deserialized: {} models", file.models.len());

            let mut traverser = SceneGraphTraverser {
                unit_size: settings.unit_size,
                scene: &file,
                models: BTreeMap::new(),
            };
            let mut root_nodes: Vec<GraphScene> = Vec::new();
            traverser.traverse(0, &mut root_nodes, IVec3::ZERO, Rotation::IDENTITY, None);
            let model_references = traverser.models;

            tracing::info!(
                "Scene graph traversed: {} models, {} root nodes",
                model_references.len(),
                root_nodes.len()
            );

            let palette_handle = load_context.add_labeled_asset("Palette", {
                let mut entries = Box::new([VoxGpuMaterial::default(); 256]);

                // Colors: dot_vox exposes the 256-entry RGBA palette as 0-based
                // slots
                for (material, color) in entries.iter_mut().zip(file.palette.iter()) {
                    material.color = U8Vec4::new(color.r, color.g, color.b, color.a);
                }
                for material in file.materials.iter() {
                    let Some(slot) = (material.id as usize).checked_sub(1) else {
                        continue;
                    };
                    if let Some(entry) = entries.get_mut(slot) {
                        entry.apply_dot_vox(material);
                    }
                }

                VoxPalette::new(self.allocator.clone(), &entries)
                    .map_err(VoxLoadingError::VulkanError)?
            });

            let model_scenes: Vec<_> = {
                // Add models
                let mut models: Vec<_> = std::mem::take(&mut file.models)
                    .into_iter()
                    .map(|a| Some(a))
                    .collect();
                let models = model_references
                    .keys()
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
                        let (tree, attribute_allocator) =
                            self.model_to_tree(model, settings.unit_size);
                        (*model_id, (tree, attribute_allocator))
                    })
                    .collect_vec_list();
                handles
                    .into_iter()
                    .flat_map(|a| a)
                    .map(|(model_id, (tree, material))| {
                        let geometry =
                            load_context.add_labeled_asset(format!("Geometry{}", model_id), tree);
                        let material = load_context
                            .add_labeled_asset(format!("Material{}", model_id), material);

                        bsn! {
                            { NameEntityReference { name: Name(format!("Model{model_id}").into()), reference: model_references[&model_id] } }
                            VoxModel {
                                geometry: {geometry},
                                material: {material},
                                palette: {palette_handle.clone()},
                            }
                        }
                    })
                    .collect()
            };

            tracing::info!("Vox scene built");
            // Wrapping the scene in a `ScenePatch` is what lets a `.vox` path be
            // spawned with the normal scene APIs. There are no scene-level asset
            // dependencies: everything it references is a labeled subasset of this
            // same asset.
            Ok(ScenePatch::load_with(
                load_context,
                bsn! {
                    Transform
                    // One root entity whose children are the file's model
                    // entities and the roots of its scene graph.
                    Children [ { model_scenes }, { root_nodes } ]
                },
            ))
        }
    }

    fn extensions(&self) -> &[&str] {
        &["vox"]
    }
}

impl VoxLoader {
    fn model_to_tree(&self, model: &dot_vox::Model, unit_size: f32) -> (VoxGeometry, VoxMaterial) {
        let mut geometry = VoxGeometry::new(self.allocator.clone(), unit_size);
        let mut material = VoxMaterial::new(self.allocator.clone());

        // Create 256x256x256 grid
        let mut accessor = geometry.tree.accessor_mut(&mut material);
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

        drop(accessor);
        // TODO: material.0.buffer_mut().flush(..);

        (geometry, material)
    }
}
