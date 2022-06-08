use std::{marker::PhantomData, sync::Arc};

use ash::vk;
use bevy_app::Plugin;
use bevy_asset::HandleId;
use bevy_ecs::{event::EventReader, prelude::FromWorld, system::ResMut};
use bevy_utils::HashMap;
use dustash::{descriptor::DescriptorVecBinding, Device};

use super::{GPURenderAsset, RenderAsset, RenderAssetStore};

/// GPU Asset that, upon creation, will be written to the corresponding bindless heap.
pub trait BindlessGPUAsset<T: RenderAsset>: GPURenderAsset<T> {
    fn binding(&self) -> dustash::descriptor::DescriptorVecBinding;
}

pub struct BindlessGPUAssetDescriptors {
    pub descriptor_vec: dustash::descriptor::DescriptorVec,
    asset_handle_to_index: HashMap<HandleId, u32>,
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

impl BindlessGPUAssetDescriptors {
    pub fn get_index_for_handle(&self, handle: impl Into<HandleId>) -> Option<u32> {
        let handle: HandleId = handle.into();
        self.asset_handle_to_index.get(&handle).map(|a| *a)
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

fn asset_binding_system<A: RenderAsset, T: BindlessGPUAsset<A>>(
    mut reader: EventReader<super::RenderAssetEvent<A>>,
    render_asset_store: RenderAssetStore<A>,
    mut store: ResMut<BindlessGPUAssetDescriptors>,
) where
    A::GPUAsset: BindlessGPUAsset<A>,
{
    let mut bindings: Vec<DescriptorVecBinding> = Vec::new();
    let mut handle_ids: Vec<HandleId> = Vec::new();
    for event in reader.iter() {
        match event {
            super::RenderAssetEvent::Created(handle) => {
                let binding = render_asset_store.get(handle).unwrap().binding();
                bindings.push(binding);
                handle_ids.push(handle.id);
            }
        }
    }

    let descriptor_ids = store.descriptor_vec.extend(bindings).unwrap();

    assert_eq!(descriptor_ids.len(), handle_ids.len());
    store
        .asset_handle_to_index
        .extend(handle_ids.into_iter().zip(descriptor_ids.into_iter()));
}
