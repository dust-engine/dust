use ash::vk;
use svo::alloc::AllocError;
pub fn map_err(err: vk::Result) -> AllocError {
    match err {
        vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => AllocError::OutOfDeviceMemory,
        vk::Result::ERROR_OUT_OF_HOST_MEMORY => AllocError::OutOfHostMemory,
        vk::Result::ERROR_MEMORY_MAP_FAILED => AllocError::MappingFailed,
        vk::Result::ERROR_TOO_MANY_OBJECTS => AllocError::TooManyObjects,
        _ => unreachable!(),
    }
}
