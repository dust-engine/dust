use std::sync::Arc;

use ash::vk;

use crate::Device;

pub struct PipelineCache {
    device: Arc<Device>,
    cache: vk::PipelineCache,
}

impl Drop for PipelineCache {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline_cache(self.cache, None);
        }
    }
}
impl PipelineCache {
    pub fn new(device: Arc<Device>) -> Self {
        let cache = unsafe {
            device
                .create_pipeline_cache(
                    &vk::PipelineCacheCreateInfo {
                        initial_data_size: 0,
                        p_initial_data: std::ptr::null(),
                        ..Default::default()
                    },
                    None,
                )
                .unwrap()
        };
        Self { device, cache }
    }
    pub fn merge(&mut self, other: impl IntoIterator<Item = PipelineCache>) {
        let caches: Vec<vk::PipelineCache> = other
            .into_iter()
            .map(|f| {
                assert!(Arc::ptr_eq(&self.device, &f.device));
                f.cache
            })
            .collect();
        unsafe {
            self.device
                .merge_pipeline_caches(self.cache, &caches)
                .unwrap()
        }
    }
    pub fn merge_one(&mut self, other: PipelineCache) {
        assert!(Arc::ptr_eq(&self.device, &other.device));
        unsafe {
            self.device
                .merge_pipeline_caches(self.cache, &[other.cache])
                .unwrap()
        }
    }
    pub fn serialize(&self) -> SerializedPipelineCache {
        SerializedPipelineCache {
            data: unsafe {
                self.device
                    .get_pipeline_cache_data(self.cache)
                    .unwrap()
                    .into_boxed_slice()
            },
        }
    }
    pub fn deserialize(device: Arc<Device>, data: SerializedPipelineCache) -> Self {
        let cache = unsafe {
            device
                .create_pipeline_cache(
                    &vk::PipelineCacheCreateInfo {
                        initial_data_size: data.data.len(),
                        p_initial_data: data.data.as_ptr() as *const std::ffi::c_void,
                        ..Default::default()
                    },
                    None,
                )
                .unwrap()
        };
        Self { device, cache }
    }
    pub unsafe fn raw(&self) -> vk::PipelineCache {
        self.cache
    }
}

pub struct SerializedPipelineCache {
    data: Box<[u8]>,
}
impl SerializedPipelineCache {
    pub fn headers(&self) -> &vk::PipelineCacheHeaderVersionOne {
        // This assumes little endian.
        let slice = &self.data[0..std::mem::size_of::<vk::PipelineCacheHeaderVersionOne>()];
        unsafe { &*(slice.as_ptr() as *const vk::PipelineCacheHeaderVersionOne) }
    }
}
