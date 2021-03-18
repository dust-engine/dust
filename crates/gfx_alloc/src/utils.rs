use gfx_hal::device as hal;
use svo::alloc::AllocError;
fn map_out_of_memory_err(err: hal::OutOfMemory) -> AllocError {
    match err {
        hal::OutOfMemory::Device => AllocError::OutOfDeviceMemory,
        hal::OutOfMemory::Host => AllocError::OutOfHostMemory,
    }
}

pub(crate) fn map_alloc_err(err: hal::AllocationError) -> AllocError {
    match err {
        hal::AllocationError::OutOfMemory(out_of_memory) => map_out_of_memory_err(out_of_memory),
        hal::AllocationError::TooManyObjects => AllocError::TooManyObjects,
    }
}

pub(crate) fn map_map_err(err: hal::MapError) -> AllocError {
    match err {
        hal::MapError::OutOfMemory(out_of_memory) => map_out_of_memory_err(out_of_memory),
        hal::MapError::MappingFailed => AllocError::MappingFailed,
        hal::MapError::Access => panic!(),
        hal::MapError::OutOfBounds => panic!(),
    }
}
