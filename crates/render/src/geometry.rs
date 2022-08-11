use std::{
    alloc::{Layout, LayoutError},
    marker::PhantomData,
    sync::Arc,
};

use crate::{
    render_asset::{GPURenderAsset, RenderAsset, RenderAssetPlugin},
    shader::SpecializedShader,
    RenderStage,
};

use bevy_app::Plugin;
use bevy_asset::{AssetServer, Handle, HandleId};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    system::{Commands, Query, SystemParam, SystemParamItem},
};

use dustash::resources::alloc::MemBuffer;

pub type GeometryAABB = ash::vk::AabbPositionsKHR;

/// The geometry defines the shape of the voxel object.
/// It serves as the "Mesh" for voxels.
///
/// SVDAG, OpenVDB, 3D Grids and ESVO could be implementors of the GeometryStructure trait.
/// Handle<Geometry> is in the world space.
pub trait Geometry: RenderAsset {
    /// The geometry represented as an array of primitives
    /// This gets persisted in the render world.

    fn aabb(&self) -> GeometryAABB;
    fn intersection_shader(asset_server: &AssetServer) -> SpecializedShader;
}

/// RenderWorld Assets.
pub trait GPUGeometry<T: Geometry>: GPURenderAsset<T> {
    fn blas_input_buffer(&self) -> &Arc<MemBuffer>;
    fn blas_input_layout() -> Layout {
        std::alloc::Layout::new::<ash::vk::AabbPositionsKHR>()
    }

    type SbtInfo: Copy;
    type GeometryInfoParams: SystemParam;
    fn geometry_info(
        &self,
        handle: &Handle<T>,
        params: &mut SystemParamItem<Self::GeometryInfoParams>,
    ) -> Self::SbtInfo;
}

/// Normalized component for Handle<Geometry>.
#[derive(Component)]
pub struct GPUGeometryPrimitives {
    pub handle: HandleId,
    pub blas_input_primitives: Option<Arc<MemBuffer>>, // None if the geometry hasn't been loaded yet.
    pub geometry_info: u64,
}

/// Insert Handle<T> in the render world for all entities with Handle<T>
fn extract_primitives<T: Geometry>(mut commands: Commands, query: Query<(Entity, &Handle<T>)>) {
    for (entity, geometry_handle) in query.iter() {
        commands
            .get_or_spawn(entity)
            .insert(geometry_handle.clone());
    }
}

pub struct GeometryPlugin<T: Geometry> {
    _marker: PhantomData<T>,
}
impl<T: Geometry> Default for GeometryPlugin<T> {
    fn default() -> Self {
        Self {
            _marker: Default::default(),
        }
    }
}

impl<T: Geometry> Plugin for GeometryPlugin<T>
where
    T::GPUAsset: GPUGeometry<T>,
{
    fn build(&self, app: &mut bevy_app::App) {
        app.add_plugin(RenderAssetPlugin::<T>::default());

        let render_app = app.sub_app_mut(crate::RenderApp);
        render_app.add_system_to_stage(RenderStage::Extract, extract_primitives::<T>);
        // TODO: maybe run prepare_primitives after prepare_geometries to decrease frame delay?
        // Right now there's a one-frame delay between geometry change and the BLAS systems see it.
    }
}
