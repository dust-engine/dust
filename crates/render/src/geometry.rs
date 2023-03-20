use bevy_reflect::TypeUuid;
use rhyolite::BufferLike;

pub enum GeometryType {
    AABBs,
    Triangles
}

pub trait Geometry: Send + Sync + 'static + TypeUuid {
    const TYPE: GeometryType;

    type BLASInputBuffer: BufferLike;
    fn blas_input_buffer(&self) -> &Self::BLASInputBuffer;
}
