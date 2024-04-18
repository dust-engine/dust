pub struct RootGeometry {}
impl rhyolite::accel_struct::BlasMarker for GeometryMarker {
    type GeometryKey = UntypedAssetId;
    type Marker = GeometryKey;
    type QueryData = &'static GeometryKey;
    type QueryFilter = ();
    type Params = Res<'static, GeometryStore>;
    fn geometry_handle(
        params: &mut Res<GeometryStore>,
        key: &bevy::ecs::query::QueryItem<Self::QueryData>,
    ) -> GeometryHandle {
        params.geometry_handle.unwrap()
    }
    fn geometry_key(
        _params: &mut Res<GeometryStore>,
        data: &bevy::ecs::query::QueryItem<Self::QueryData>,
    ) -> Self::GeometryKey {
        data.asset_id
    }
}

pub trait Geometry: Send + Sync + 'static + Asset + TypePath {
    const TYPE: GeometryType;

    type BLASInputBufferFuture: GPUCommandFuture<Output = Arc<ResidentBuffer>>;
    fn blas_input_buffer(&self) -> Self::BLASInputBufferFuture;

    fn geometry_flags(&self) -> vk::GeometryFlagsKHR {
        vk::GeometryFlagsKHR::OPAQUE
    }

    /// Layout for one single AABB entry
    fn layout(&self) -> Layout {
        Layout::new::<vk::AabbPositionsKHR>()
    }
}
