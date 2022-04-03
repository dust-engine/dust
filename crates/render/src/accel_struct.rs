use std::{pin::Pin, sync::Arc};

use ash::{prelude::VkResult, vk};
use bevy_app::Plugin;
use bevy_asset::HandleUntyped;
use bevy_ecs::prelude::FromWorld;
use bevy_utils::{HashMap, HashSet};
use dustash::{
    command::pool::CommandPool,
    queue::{QueueType, Queues},
    Device,
};
use futures_lite::Future;

// One global for all geometry
pub struct AccelerationStructureStore {
    /// Mapping from Geometry handle to Acceleration Structure
    pub(crate) accel_structs:
        HashMap<HandleUntyped, Arc<dustash::accel_struct::AccelerationStructure>>,

    /// Acceleration Structures currently being built
    pub(crate) pending_accel_structs:
        HashMap<HandleUntyped, Arc<dustash::accel_struct::AccelerationStructure>>,

    /// When this is Some, there are acceleration strucutres currently being built.
    /// When this resolves, the acceleration structure builds are completed.
    pub(crate) accel_structs_build_completion:
        Option<Pin<Box<dyn Future<Output = VkResult<()>> + Send + Sync>>>,

    /// Geometries waiting to have their BLAS built
    pub(crate) queued_accel_structs: HashSet<HandleUntyped>,

    pub(crate) transfer_pool: Arc<CommandPool>,
    pub(crate) compute_pool: Arc<CommandPool>,
}

impl FromWorld for AccelerationStructureStore {
    fn from_world(world: &mut bevy_ecs::prelude::World) -> Self {
        let device = world.get_resource::<Arc<Device>>().unwrap();
        let queues = world.get_resource::<Queues>().unwrap();
        AccelerationStructureStore {
            accel_structs: HashMap::new(),
            pending_accel_structs: HashMap::new(),
            accel_structs_build_completion: None,
            queued_accel_structs: HashSet::new(),
            transfer_pool: Arc::new(
                CommandPool::new(
                    device.clone(),
                    vk::CommandPoolCreateFlags::TRANSIENT,
                    queues.of_type(QueueType::Graphics).family_index(),
                )
                .unwrap(),
            ),
            compute_pool: Arc::new(
                CommandPool::new(
                    device.clone(),
                    vk::CommandPoolCreateFlags::TRANSIENT,
                    queues.of_type(QueueType::Compute).family_index(),
                )
                .unwrap(),
            ),
        }
    }
}

#[derive(Default)]
pub struct AccelerationStructurePlugin;
impl Plugin for AccelerationStructurePlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.init_resource::<AccelerationStructureStore>();
    }
}
