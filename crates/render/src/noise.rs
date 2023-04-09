use bevy_app::Plugin;
use bevy_asset::AssetServer;
use bevy_ecs::world::FromWorld;
use rhyolite_bevy::SlicedImageArray;

#[derive(bevy_ecs::system::Resource)]
pub struct BlueNoise {
    pub scalar: bevy_asset::Handle<SlicedImageArray>,
    pub vec3: bevy_asset::Handle<SlicedImageArray>,
    pub unitvec3: bevy_asset::Handle<SlicedImageArray>,
    pub unitvec3_cosine: bevy_asset::Handle<SlicedImageArray>,
}

impl FromWorld for BlueNoise {
    fn from_world(world: &mut bevy_ecs::world::World) -> Self {
        let asset_server = world.resource::<AssetServer>();
        BlueNoise {
            scalar: asset_server.load("stbn_scalar_2Dx1Dx1D_128x128x64x1.png"),
            vec3: asset_server.load("stbn_vec3_2Dx1D_128x128x64.png"),
            unitvec3: asset_server.load("stbn_unitvec3_2Dx1D_128x128x64.png"),
            unitvec3_cosine: asset_server.load("stbn_unitvec3_cosine_2Dx1D_128x128x64.png"),
        }
    }
}
