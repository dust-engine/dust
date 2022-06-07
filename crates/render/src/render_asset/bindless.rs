use std::{marker::PhantomData, sync::Arc};

use ash::vk;
use bevy_app::Plugin;
use bevy_asset::HandleId;
use bevy_ecs::prelude::FromWorld;
use bevy_utils::HashMap;
use dustash::Device;

use super::{GPURenderAsset, RenderAsset};

/// GPU Asset that, upon creation, will be written to the corresponding bindless heap.
pub trait BindlessGPUAsset<T: RenderAsset>: GPURenderAsset<T> {
    fn binding(&self) -> dustash::descriptor::DescriptorVecBinding;
}

pub struct BindlessGPUAssetDescriptors {
    pub descriptor_vec: dustash::descriptor::DescriptorVec,
    pub asset_handle_to_index: HashMap<HandleId, u32>,
}
impl FromWorld for BindlessGPUAssetDescriptors {
    fn from_world(world: &mut bevy_ecs::prelude::World) -> Self {
        let device: &Arc<Device> = world.resource();
        Self {
            descriptor_vec: dustash::descriptor::DescriptorVec::new(
                device.clone(),
                vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            )
            .unwrap(),
            asset_handle_to_index: HashMap::new(),
        }
    }
}

pub struct BindlessGPUAssetPlugin<R: RenderAsset, T: BindlessGPUAsset<R>> {
    _marker: PhantomData<(R, T)>,
}

impl<R: RenderAsset, T: BindlessGPUAsset<R>> Plugin for BindlessGPUAssetPlugin<R, T> {
    fn build(&self, app: &mut bevy_app::App) {
        if app
            .world
            .get_resource::<BindlessGPUAssetDescriptors>()
            .is_none()
        {
            app.world.init_resource::<BindlessGPUAssetDescriptors>();
        }
    }
}

fn asset_binding_system() {
    todo!("Allocate index and bind upon resource creation.")
}
