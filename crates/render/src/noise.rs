use bevy_asset::{AssetServer, Assets};
use bevy_ecs::world::FromWorld;
use rhyolite::ash::vk;
use rhyolite_bevy::SlicedImageArray;

#[derive(bevy_ecs::system::Resource)]
pub struct BlueNoise {
    pub scalar: bevy_asset::Handle<SlicedImageArray>,
    pub vec2: bevy_asset::Handle<SlicedImageArray>,
    pub unitvec2: bevy_asset::Handle<SlicedImageArray>,
    pub vec3: bevy_asset::Handle<SlicedImageArray>,
    pub unitvec3: bevy_asset::Handle<SlicedImageArray>,
    pub unitvec3_cosine: bevy_asset::Handle<SlicedImageArray>,
}

impl FromWorld for BlueNoise {
    fn from_world(world: &mut bevy_ecs::world::World) -> Self {
        let asset_server = world.resource::<AssetServer>();
        BlueNoise {
            scalar: asset_server.load("stbn_scalar_2Dx1Dx1D_128x128x64x1.png"),
            vec2: asset_server.load("stbn_vec2_2Dx1D_128x128x64.png"),
            unitvec2: asset_server.load("stbn_unitvec2_2Dx1D_128x128x64.png"),
            vec3: asset_server.load("stbn_vec3_2Dx1D_128x128x64.png"),
            unitvec3: asset_server.load("stbn_unitvec3_2Dx1D_128x128x64.png"),
            unitvec3_cosine: asset_server.load("stbn_unitvec3_cosine_2Dx1D_128x128x64.png"),
        }
    }
}

impl BlueNoise {
    pub fn as_descriptors(
        &self,
        image_arrays: &Assets<SlicedImageArray>,
        index: u32,
    ) -> Option<[vk::DescriptorImageInfo; 6]> {
        use rhyolite::{ImageLike, ImageViewExt};
        let mut descriptors: [vk::DescriptorImageInfo; 6] = [Default::default(); 6];
        let handles = [
            &self.scalar,
            &self.vec2,
            &self.unitvec2,
            &self.vec3,
            &self.unitvec3,
            &self.unitvec3_cosine,
        ];
        for (i, (desc, handle)) in descriptors.iter_mut().zip(handles.iter()).enumerate() {
            let Some(img) = image_arrays.get(*handle) else {
                return None;
            };
            let noise_texture_index = index % img.subresource_range().layer_count;
            *desc = img
                .slice(noise_texture_index as usize)
                .as_descriptor(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        }
        Some(descriptors)
    }
}
