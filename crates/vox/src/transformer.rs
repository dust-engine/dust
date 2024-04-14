//！Converting dotvox models into vdb tree
use bevy::{asset::transformer::{AssetTransformer, TransformedAsset}, math::UVec3};

use crate::{VoxModel, VoxTree};

#[derive(Debug, thiserror::Error)]
pub enum VoxTransformerError {
    #[error("parse error: {0}")]
    ParseError(&'static str),
}


pub struct VoxTransformer;
impl AssetTransformer for VoxTransformer {
    type AssetInput = VoxModel;

    type AssetOutput = VoxTree;

    type Settings = ();

    type Error = VoxTransformerError;

    fn transform<'a>(
        &'a self,
        asset: TransformedAsset<Self::AssetInput>,
        _settings: &'a Self::Settings,
    ) -> bevy::utils::BoxedFuture<'a, Result<TransformedAsset<Self::AssetOutput>, Self::Error>> {
        Box::pin(async {
            let mut tree = crate::Tree::new();
            let mut accessor = tree.accessor_mut();
            let size_y = asset.get().size.y;
            for voxel in asset.get().voxels.iter() {
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

                accessor.set(coords, Some(true));
            }
            Ok(asset.replace_asset(VoxTree(tree)))
        })
    }
}
