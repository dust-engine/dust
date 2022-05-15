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
            let geometry = super::AABBGeometry {
                primitives: aabbs.to_owned().into_boxed_slice(),
            };
            load_context.set_default_asset(LoadedAsset::new(geometry));
            Ok(())
        })
    }

    fn extensions(&self) -> &[&str] {
        &["aabb"]
    }
}
