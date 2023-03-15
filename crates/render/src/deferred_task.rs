use std::sync::Arc;

use bevy_tasks::AsyncComputeTaskPool;
use once_cell::sync::OnceCell;
use rhyolite::{
    ash::{prelude::VkResult, vk},
    DeferredOperation,
};

static DEFERRED_TASK_POOL: OnceCell<DeferredTaskPool> = OnceCell::new();
pub struct DeferredTaskPool {
    device: Arc<rhyolite::Device>,
    dho: Arc<rhyolite::DeferredOperationTaskPool>,
    compute_pool: &'static bevy_tasks::AsyncComputeTaskPool,
}

impl DeferredTaskPool {
    pub fn init(device: Arc<rhyolite::Device>) {
        DEFERRED_TASK_POOL.get_or_init(|| Self {
            device: device.clone(),
            dho: Arc::new(rhyolite::DeferredOperationTaskPool::new(device)),
            compute_pool: AsyncComputeTaskPool::get(),
        });
    }
    pub fn get() -> &'static Self {
        DEFERRED_TASK_POOL.get().expect(
            "A DeferredTaskPool has not been initialized yet. Please call \
            DeferredTaskPool::init beforehand.",
        )
    }
    pub fn schedule<T: Send + 'static>(
        &self,
        op: impl FnOnce(Option<&mut DeferredOperation>) -> (T, vk::Result) + Send + 'static,
    ) -> bevy_tasks::Task<VkResult<T>> {
        let mut deferred_operation = DeferredOperation::new(self.device.clone()).ok();
        let dho = self.dho.clone();
        self.compute_pool.spawn(async move {
            let (value, code) = op(deferred_operation.as_mut());
            match code {
                vk::Result::SUCCESS => return Ok(value),
                vk::Result::OPERATION_DEFERRED_KHR => {
                    let deferred_operation = deferred_operation.unwrap();
                    let result = dho.schedule_deferred_operation(deferred_operation).await;
                    return result.result_with_success(value);
                }
                vk::Result::OPERATION_NOT_DEFERRED_KHR => {
                    return Ok(value);
                }
                other => return Err(other),
            }
        })
    }
}

pub enum DeferredValue<T> {
    Pending(bevy_tasks::Task<VkResult<T>>),
    Done(T),
    Errored(vk::Result),
    None,
}
impl<T> DeferredValue<T> {
    pub fn is_done(&self) -> bool {
        match self {
            Self::Done(_) => true,
            _ => false,
        }
    }
    pub fn map<R>(&self, mapper: impl FnOnce(&T) -> R) -> Option<R> {
        match self {
            Self::Done(value) => Some(mapper(value)),
            _ => None,
        }
    }
    pub fn try_get(&mut self) -> Option<&mut T> {
        match self {
            Self::Pending(task) => {
                if task.is_finished() {
                    let task = match std::mem::replace(self, Self::None) {
                        Self::Pending(task) => task,
                        _ => unreachable!(),
                    };
                    let value = futures_lite::future::block_on(task);
                    match value {
                        Ok(value) => {
                            *self = Self::Done(value);
                            match self {
                                Self::Done(value) => return Some(value),
                                _ => unreachable!(),
                            }
                        }
                        Err(result) => {
                            *self = Self::Errored(result);
                            return None;
                        }
                    }
                } else {
                    return None;
                }
            }
            Self::Done(value) => return Some(value),
            Self::Errored(_) => return None,
            Self::None => unreachable!(),
        }
    }
}
impl<T> From<bevy_tasks::Task<VkResult<T>>> for DeferredValue<T> {
    fn from(task: bevy_tasks::Task<VkResult<T>>) -> Self {
        DeferredValue::Pending(task)
    }
}
