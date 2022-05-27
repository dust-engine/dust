use bevy_asset::{AssetLoader, LoadedAsset};

#[derive(Default)]
pub struct ExplicitAABBPrimitivesLoader;
impl AssetLoader for ExplicitAABBPrimitivesLoader {
    fn load<'a>(
        &'a self,
        bytes: &'a [u8],
        load_context: &'a mut bevy_asset::LoadContext,
    ) -> bevy_asset::BoxedFuture<'a, Result<(), anyhow::Error>> {
        println!("Loaded AABB File");
        Box::pin(async {
            let num_primitives = bytes.len() / std::mem::size_of::<ash::vk::AabbPositionsKHR>();
            let aabbs = unsafe {
                std::slice::from_raw_parts(
                    bytes.as_ptr() as *const ash::vk::AabbPositionsKHR,
                    num_primitives,
                )
            };
            let mut geometry = super::AABBGeometry {
                primitives: aabbs.to_owned().into_boxed_slice(),
            };
            let mut total_aabb = ash::vk::AabbPositionsKHR {
                min_x: f32::INFINITY,
                min_y: f32::INFINITY,
                min_z: f32::INFINITY,
                max_x: -f32::INFINITY,
                max_y: -f32::INFINITY,
                max_z: -f32::INFINITY,
            };
            for primitive in geometry.primitives.iter_mut() {
                primitive.min_x *= 0.05;
                primitive.min_y *= 0.05;
                primitive.min_z *= 0.05;
                primitive.max_x *= 0.05;
                primitive.max_y *= 0.05;
                primitive.max_z *= 0.05;
                total_aabb.min_x = total_aabb.min_x.min(primitive.min_x);
                total_aabb.min_y = total_aabb.min_y.min(primitive.min_y);
                total_aabb.min_z = total_aabb.min_z.min(primitive.min_z);
                total_aabb.max_x = total_aabb.max_x.max(primitive.max_x);
                total_aabb.max_y = total_aabb.max_y.max(primitive.max_y);
                total_aabb.max_z = total_aabb.max_z.max(primitive.max_z);
            }
            let mid_x = (total_aabb.min_x + total_aabb.max_x) / 2.0;
            let mid_y = (total_aabb.min_y + total_aabb.max_y) / 2.0;
            let mid_z = (total_aabb.min_z + total_aabb.max_z) / 2.0;
            for primitive in geometry.primitives.iter_mut() {
                primitive.min_x -= mid_x;
                primitive.min_y -= mid_y;
                primitive.min_z -= mid_z;
                primitive.max_x -= mid_x;
                primitive.max_y -= mid_y;
                primitive.max_z -= mid_z;
            }
            load_context.set_default_asset(LoadedAsset::new(geometry));
            Ok(())
        })
    }

    fn extensions(&self) -> &[&str] {
        &["aabb"]
    }
}
