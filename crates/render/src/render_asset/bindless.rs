use std::{marker::PhantomData, sync::Arc};

use ash::vk;
use bevy_app::Plugin;
use bevy_asset::HandleId;
use bevy_ecs::{
    event::EventReader,
    prelude::FromWorld,
    system::{Res, ResMut},
};
use bevy_utils::HashMap;
use dustash::{descriptor::DescriptorVecBinding, Device};

use crate::{RenderApp, RenderStage};

use super::{GPURenderAsset, RenderAsset, RenderAssetStore, PrepareRenderAssetsSystem};
use bevy_ecs::schedule::ParallelSystemDescriptorCoercion;

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

pub struct BindlessGPUAssetPlugin<R: RenderAsset> {
    _marker: PhantomData<R>,
}

impl<R: RenderAsset> Default for BindlessGPUAssetPlugin<R> {
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<R: RenderAsset> Plugin for BindlessGPUAssetPlugin<R>
where
    R::GPUAsset: BindlessGPUAsset<R>,
{
    fn build(&self, app: &mut bevy_app::App) {
        if app
            .world
            .get_resource::<BindlessGPUAssetDescriptors>()
            .is_none()
        {
            app.world.init_resource::<BindlessGPUAssetDescriptors>();
        }
        app.sub_app_mut(RenderApp)
            .add_system_to_stage(RenderStage::Prepare, asset_binding_system::<R>.after(PrepareRenderAssetsSystem::<R>::default()));
    }
}

fn asset_binding_system<A: RenderAsset>(
    mut reader: EventReader<super::RenderAssetEvent<A>>,
    render_asset_store: Res<RenderAssetStore<A>>,
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
    if bindings.len() == 0 {
        return;
    }

    let descriptor_ids = store.descriptor_vec.extend(bindings).unwrap();
    println!("Bindless returned descreiptor ids {:?}", descriptor_ids);

    assert_eq!(descriptor_ids.len(), handle_ids.len());
    store
        .asset_handle_to_index
        .extend(handle_ids.into_iter().zip(descriptor_ids.into_iter()));
}
