pub mod loader;
use bevy_app::Plugin;
use bevy_asset::{AddAsset, AssetServer, Handle};
use bevy_reflect::TypeUuid;
use dust_render::geometry::{GPUGeometry, Geometry};
use dustash::{command::recorder::CommandRecorder, ray_tracing::sbt::SpecializationInfo};
use loader::ExplicitAABBPrimitivesLoader;

#[derive(TypeUuid)]
#[uuid = "75a9a733-04d7-4abb-8600-9a7d24ff0597"]
pub struct AABBGeometry {
    primitives: Box<[ash::vk::AabbPositionsKHR]>,
}

pub struct AABBGPUGeometry {}

impl Geometry for AABBGeometry {
    type GPUGeometry = AABBGPUGeometry;

    type ChangeSet = ();

    type BuildSet = ();

    fn aabb(&self) -> dust_render::geometry::GeometryAABB {
        todo!()
    }

    fn intersection_shader(asset_server: &AssetServer) -> Handle<dust_render::shader::Shader> {
        todo!()
    }

    fn specialization() -> SpecializationInfo {
        todo!()
    }

    fn generate_builds(&mut self) -> Self::BuildSet {
        todo!()
    }

    fn emit_changes(&mut self) -> Self::ChangeSet {
        todo!()
    }
}

impl GPUGeometry<AABBGeometry> for AABBGPUGeometry {
    fn build(build_set: <AABBGeometry as Geometry>::BuildSet) -> Self {
        todo!()
    }

    fn apply_change_set(&mut self, change_set: <AABBGeometry as Geometry>::ChangeSet) {
        todo!()
    }
}

#[derive(Default)]
pub struct ExplicitAABBPlugin;
impl Plugin for ExplicitAABBPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.add_plugin(dust_render::geometry::GeometryPlugin::<AABBGeometry>::default())
            .add_asset_loader(ExplicitAABBPrimitivesLoader::default());
    }
}
