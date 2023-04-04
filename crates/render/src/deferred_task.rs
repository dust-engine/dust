use std::sync::Arc;

use once_cell::sync::OnceCell;
use rhyolite::ash::{prelude::VkResult, vk};

static DEFERRED_TASK_POOL: OnceCell<DeferredTaskPool> = OnceCell::new();
pub struct DeferredTaskPool {
    dho: Arc<rhyolite::DeferredOperationTaskPool>,
}

impl DeferredTaskPool {
    pub fn init(device: Arc<rhyolite::Device>) {
        DEFERRED_TASK_POOL.get_or_init(|| Self {
            dho: Arc::new(rhyolite::DeferredOperationTaskPool::new(device)),
        });
    }
    pub fn get() -> &'static Self {
        DEFERRED_TASK_POOL.get().expect(
            "A DeferredTaskPool has not been initialized yet. Please call \
            DeferredTaskPool::init beforehand.",
        )
    }
    pub fn inner(&self) -> &Arc<rhyolite::DeferredOperationTaskPool> {
        &self.dho
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
            Self::Pending(task) => task.is_finished(),
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
