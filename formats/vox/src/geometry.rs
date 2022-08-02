use crate::material::{DummyMaterial, GPUDummyMaterial};
use crate::vox_loader::*;
use bevy_asset::AddAsset;
use dust_render::{
    geometry::{GPUGeometry, Geometry},
    render_asset::{GPURenderAsset, RenderAsset},
};
use glam::Vec3;

pub type VoxGeometryInner = dust_format_vdb::VdbGeometry<TreeRoot>;

#[derive(bevy_reflect::TypeUuid)]
#[uuid = "4b169454-eb57-446f-adc3-b7c409f60f5b"]
/// Wrapper for VoxGeometryInner. Without the wrapper, the linking fails due to a rustc bug
pub struct VoxGeometry(VoxGeometryInner);

impl VoxGeometry {
    pub fn new(tree: Tree, unit_size: Vec3) -> Self {
        let geometry = VoxGeometryInner::new(tree, unit_size);
        Self(geometry)
    }
}

impl RenderAsset for VoxGeometry {
    type GPUAsset = VoxGPUGeometry;

    type BuildData = <VoxGeometryInner as RenderAsset>::BuildData;

    type CreateBuildDataParam = <VoxGeometryInner as RenderAsset>::CreateBuildDataParam;

    fn create_build_data(
        &mut self,
        param: &mut bevy_ecs::system::SystemParamItem<Self::CreateBuildDataParam>,
    ) -> Self::BuildData {
        self.0.create_build_data(param)
    }
}

impl Geometry for VoxGeometry {
    fn aabb(&self) -> dust_render::geometry::GeometryAABB {
        self.0.aabb()
    }

    fn intersection_shader(
        asset_server: &bevy_asset::AssetServer,
    ) -> dust_render::shader::SpecializedShader {
        VoxGeometryInner::intersection_shader(asset_server)
    }
}

type VoxGPUGeometryInner = <VoxGeometryInner as RenderAsset>::GPUAsset;
pub struct VoxGPUGeometry(VoxGPUGeometryInner);
impl GPURenderAsset<VoxGeometry> for VoxGPUGeometry {
    type BuildParam = <VoxGPUGeometryInner as GPURenderAsset<VoxGeometryInner>>::BuildParam;

    fn build(
        build_set: <VoxGeometry as RenderAsset>::BuildData,
        commands_future: &mut dustash::sync::CommandsFuture,
        params: &mut bevy_ecs::system::SystemParamItem<Self::BuildParam>,
    ) -> Self {
        let geometry = <VoxGPUGeometryInner as GPURenderAsset<VoxGeometryInner>>::build(
            build_set,
            commands_future,
            params,
        );
        Self(geometry)
    }
}

impl GPUGeometry<VoxGeometry> for VoxGPUGeometry {
    fn blas_input_buffer(&self) -> &std::sync::Arc<dustash::resources::alloc::MemBuffer> {
        GPUGeometry::<VoxGeometryInner>::blas_input_buffer(&self.0)
    }

    type SbtInfo = <VoxGPUGeometryInner as GPUGeometry<VoxGeometryInner>>::SbtInfo;

    type GeometryInfoParams =
        <VoxGPUGeometryInner as GPUGeometry<VoxGeometryInner>>::GeometryInfoParams;

    fn geometry_info(
        &self,
        handle: &bevy_asset::Handle<VoxGeometry>,
        params: &mut bevy_ecs::system::SystemParamItem<Self::GeometryInfoParams>,
    ) -> Self::SbtInfo {
        GPUGeometry::<VoxGeometryInner>::geometry_info(
            &self.0,
            unsafe { std::mem::transmute(handle) },
            params,
        )
    }
}
