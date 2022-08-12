use bevy_ecs::{
    prelude::Bundle,
    world::{EntityMut, World},
};
use bevy_hierarchy::{BuildWorldChildren, WorldChildBuilder};
use bevy_transform::prelude::{GlobalTransform, Transform};
use dot_vox::{Color, DotVoxData, Model, SceneNode, SignedPermutationMatrix};
use dust_vdb::hierarchy;
use glam::{IVec3, UVec3};
/// MagicaVoxel trees are 256x256x256 max, so the numbers in the
/// hierarchy must sum up to 8 where 2^8 = 256.

pub type TreeRoot = hierarchy!(4, 2, 2);
pub type Tree = dust_vdb::Tree<TreeRoot>;
use crate::{palette::VoxPalette, VoxBundle, VoxGeometry};

use bevy_asset::{AssetLoader, Assets, Handle, LoadedAsset};

use crate::material::PaletteMaterial;

#[derive(Default)]
pub struct VoxLoader {}

struct SceneGraphTraverser<'a, 'd> {
    unit_size: f32,
    scene: &'a DotVoxData,
    cache: &'a mut Vec<Option<(Handle<VoxGeometry>, Handle<PaletteMaterial>)>>,
    palette: &'a Handle<VoxPalette>,
    load_context: &'a mut bevy_asset::LoadContext<'d>,
}

impl<'a, 'd> SceneGraphTraverser<'a, 'd> {
    fn traverse_recursive(
        &mut self,
        node: u32,
        parent: WorldOrParent<'_, '_>,
        translation: glam::IVec3,
        rotation: SignedPermutationMatrix,
    ) {
        let node = &self.scene.scenes[node as usize];
        match node {
            SceneNode::Transform {
                attributes: _,
                frames,
                child,
            } => {
                if frames.len() != 1 {
                    unimplemented!("Multiple frame in transform node");
                }
                let frame = &frames[0];
                let this_translation = frame.translation().map(IVec3::from).unwrap_or(IVec3::ZERO);

                let this_rotation = frame
                    .rotation()
                    .unwrap_or(SignedPermutationMatrix::IDENTITY);
                //let rotation = rotation * this_rotation; // reverse?
                let translation = translation + this_translation;

                self.traverse_recursive(*child, parent, translation, this_rotation);
            }
            SceneNode::Group {
                attributes: _,
                children,
            } => {
                parent
                    .spawn()
                    .insert(self.to_transform(translation, rotation))
                    .insert(GlobalTransform::default())
                    .with_children(|builder| {
                        for &i in children {
                            self.traverse_recursive(
                                i,
                                WorldOrParent::Parent(builder),
                                glam::IVec3::ZERO,
                                SignedPermutationMatrix::IDENTITY,
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

                let (geometry_handle, material_handle) = if let Some((geometry, material)) =
                    &self.cache[shape_model.model_id as usize]
                {
                    // Retrieve from cache
                    println!("Reused bundle {}", shape_model.model_id);
                    (geometry.clone(), material.clone())
                } else {
                    // Build new
                    println!("Spawned bundle {}", shape_model.model_id);
                    let model = &self.scene.models[shape_model.model_id as usize];
                    if model.voxels.len() == 0 {
                        return;
                    }
                    let (tree, material) = self.load_model(model, self.palette);
                    let geometry = VoxGeometry::from_tree(tree, 1.0);

                    let geometry_handle = self.load_context.set_labeled_asset(
                        &format!("Geometry{}", shape_model.model_id),
                        LoadedAsset::new(geometry),
                    );
                    let material_handle = self.load_context.set_labeled_asset(
                        &format!("Material{}", shape_model.model_id),
                        LoadedAsset::new(material),
                    );
                    self.cache[shape_model.model_id as usize] =
                        Some((geometry_handle.clone(), material_handle.clone()));
                    (geometry_handle, material_handle)
                };

                parent.spawn_bundle(VoxBundle {
                    transform: self.to_transform(translation, rotation),
                    ..VoxBundle::from_geometry_material(geometry_handle, material_handle)
                });
            }
        }
    }

    fn to_transform(
        &self,
        translation: glam::IVec3,
        rotation: SignedPermutationMatrix,
    ) -> Transform {
        let (quat, scale) = rotation.to_quat_scale();
        Transform {
            translation: glam::Vec3::new(
                translation.x as f32,
                translation.y as f32,
                translation.z as f32,
            ),
            rotation: glam::Quat::from_array(quat),
            scale: glam::Vec3::from_array(scale),
        }
    }

    fn load_model(&self, model: &Model, palette: &Handle<VoxPalette>) -> (Tree, PaletteMaterial) {
        let mut palette_index_collector = crate::collector::ModelIndexCollector::new();

        let mut tree = Tree::new();
        for voxel in model.voxels.iter() {
            let mut voxel = voxel.clone();
            std::mem::swap(&mut voxel.z, &mut voxel.y);
            voxel.z = 255 - voxel.z;

            let coords: UVec3 = UVec3 {
                x: voxel.x as u32,
                y: voxel.y as u32,
                z: voxel.z as u32,
            };
            tree.set_value(coords, Some(true));
            palette_index_collector.set(voxel.clone());
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
        let material_data: Vec<u8> = palette_indexes.collect();
        (tree, PaletteMaterial::new(palette.clone(), material_data))
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
            let mut cache: Vec<_> = vec![None; file.models.len()];
            let mut traverser = SceneGraphTraverser {
                unit_size: 1.0,
                scene: &file,
                cache: &mut cache,
                palette: &palette_handle,
                load_context,
            };
            traverser.traverse_recursive(
                0,
                WorldOrParent::World(&mut world),
                IVec3::ZERO,
                SignedPermutationMatrix::IDENTITY,
            );
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
    fn spawn(self) -> EntityMut<'w> {
        match self {
            WorldOrParent::World(world) => world.spawn(),
            WorldOrParent::Parent(parent) => parent.spawn(),
        }
    }
    fn spawn_bundle(self, bundle: impl Bundle + Send + Sync + 'static) -> EntityMut<'w> {
        match self {
            WorldOrParent::World(world) => {
                let mut a = world.spawn();
                a.insert_bundle(bundle);
                a
            }
            WorldOrParent::Parent(parent) => parent.spawn_bundle(bundle),
        }
    }
}
