pub mod loader;
use bevy_asset::{AssetServer, Handle, AddAsset};
use bevy_reflect::TypeUuid;
use dustash::{command::recorder::CommandRecorder, ray_tracing::sbt::SpecializationInfo};
use loader::ExplicitAABBPrimitivesLoader;

#[derive(TypeUuid)]
#[uuid = "75a9a733-04d7-4abb-8600-9a7d24ff0597"]
pub struct AABBGeometry {
    primitives: Box<[ash::vk::AabbPositionsKHR]>,
}

pub struct AABBGeometryGPUAsset {}
impl dust_render::geometry::GeometryPrimitiveArray for AABBGeometryGPUAsset {
    fn rebuild_blas(
        &self,
        _command_recorder: &mut CommandRecorder,
    ) -> dustash::accel_struct::AccelerationStructure {
        todo!()
    }
}

pub enum AABBGeometryChangeSet {
    Rebuild(Box<[ash::vk::AabbPositionsKHR]>),
    None,
}
impl dust_render::geometry::GeometryChangeSet<AABBGeometryGPUAsset> for AABBGeometryChangeSet {
    type Param = ();

    fn into_primitives(
        self,
        _command_recorder: &mut CommandRecorder,
        _params: &mut bevy_ecs::system::SystemParamItem<Self::Param>,
    ) -> (
        AABBGeometryGPUAsset,
        Vec<dust_render::geometry::GeometryBLASBuildDependency>,
    ) {
        todo!()
    }

    fn apply_on(
        self,
        _primitives: &mut AABBGeometryGPUAsset,
        _command_recorder: &mut CommandRecorder,
        _params: &mut bevy_ecs::system::SystemParamItem<Self::Param>,
    ) -> Option<Vec<dust_render::geometry::GeometryBLASBuildDependency>> {
        todo!()
    }
}

impl dust_render::geometry::Geometry for AABBGeometry {
    type Primitives = AABBGeometryGPUAsset;

    type ChangeSet = AABBGeometryChangeSet;

    fn aabb(&self) -> dust_render::geometry::GeometryAABB {
        todo!()
    }

    fn intersection_shader(_asset_server: &AssetServer) -> Handle<dust_render::shader::Shader> {
        todo!()
    }

    fn specialization() -> SpecializationInfo {
        todo!()
    }

    fn generate_changes(&self) -> Self::ChangeSet {
        todo!()
    }
}

#[derive(Default)]
pub struct ExplicitAABBPrimitivesPlugin;
impl bevy_app::Plugin for ExplicitAABBPrimitivesPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.add_plugin(dust_render::geometry::GeometryPlugin::<AABBGeometry>::default())
            .add_asset_loader(loader::ExplicitAABBPrimitivesLoader::default());
    }
}
