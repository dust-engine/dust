use crate::back as back;
use crate::hal as hal;
use hal::prelude::*;
use std::alloc::Layout;

pub struct SharedBuffer<'a> {
    alignment: usize,
    current_size: usize,
    integrated: bool,
    device: &'a <back::Backend as hal::Backend>::Device,
    mem: <back::Backend as hal::Backend>::Memory,
    staging_mem: Option<<back::Backend as hal::Backend>::Memory>,
}

impl<'a> SharedBuffer<'a> {
    pub unsafe fn alloc_buffer(&mut self, data: &mut [u8], usage: hal::buffer::Usage)
        -> Result<<back::Backend as hal::Backend>::Buffer, hal::buffer::CreationError> {
        let buffer_layout = Layout::for_value(data)
            .align_to(self.alignment)
            .unwrap();
        let layout = buffer_layout.pad_to_align();
        let buffer = self.device.create_buffer(
            layout.size() as u64,
            usage,
            hal::memory::SparseFlags::empty()
        )?;

        todo!()
    }
}
