use ash::prelude::VkResult;
use ash::vk;

use std::ffi::CStr;
use std::fmt::Debug;
use std::ops::Deref;
use std::sync::Arc;

use crate::descriptor::DescriptorSetLayout;

use crate::{Device, HasDevice};

pub struct SpirvShader<T: Deref<Target = [u32]>> {
    pub data: T,
}

impl<T: Deref<Target = [u32]>> SpirvShader<T> {
    pub fn build(self, device: Arc<Device>) -> VkResult<ShaderModule> {
        ShaderModule::new(device, &self.data)
    }
}

pub struct ShaderModule {
    device: Arc<Device>,
    module: vk::ShaderModule,
}
impl Debug for ShaderModule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("ShaderModule").field(&self.module).finish()
    }
}
impl ShaderModule {
    pub fn new(device: Arc<Device>, data: &[u32]) -> VkResult<Self> {
        let module = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo {
                    code_size: std::mem::size_of_val(data),
                    p_code: data.as_ptr(),
                    ..Default::default()
                },
                None,
            )
        }?;
        Ok(Self { device, module })
    }
    pub fn raw(&self) -> vk::ShaderModule {
        self.module
    }
    pub fn specialized<'a>(
        &'a self,
        entry_point: &'a CStr,
        stage: vk::ShaderStageFlags,
    ) -> SpecializedShader<&'a ShaderModule> {
        SpecializedShader {
            stage,
            flags: vk::PipelineShaderStageCreateFlags::empty(),
            shader: self,
            specialization_info: Default::default(),
            entry_point,
        }
    }
}

impl HasDevice for ShaderModule {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}
impl Drop for ShaderModule {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_shader_module(self.module, None);
        }
    }
}

#[derive(Clone)]
pub struct ShaderModuleEntryPoint {
    pub stage: vk::ShaderStageFlags,
    pub desc_sets: Vec<Arc<DescriptorSetLayout>>,
    pub push_constant_range: Option<vk::PushConstantRange>,
}

#[derive(Clone, Default, Debug)]
pub struct SpecializationInfo {
    pub(super) data: Vec<u8>,
    pub(super) entries: Vec<vk::SpecializationMapEntry>,
}
impl SpecializationInfo {
    pub unsafe fn raw_info(&self) -> vk::SpecializationInfo {
        vk::SpecializationInfo {
            map_entry_count: self.entries.len() as u32,
            p_map_entries: if self.entries.is_empty() {
                std::ptr::null()
            } else {
                self.entries.as_ptr()
            },
            data_size: self.data.len(),
            p_data: if self.data.is_empty() {
                std::ptr::null()
            } else {
                self.data.as_ptr() as *const _
            },
        }
    }
    pub const fn new() -> Self {
        Self {
            data: Vec::new(),
            entries: Vec::new(),
        }
    }
    pub fn push<T: Copy + 'static>(&mut self, constant_id: u32, item: T) {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<bool>() {
            unsafe {
                let value: bool = std::mem::transmute_copy(&item);
                self.push_bool(constant_id, value);
                return;
            }
        }
        let size = std::mem::size_of::<T>();
        self.entries.push(vk::SpecializationMapEntry {
            constant_id,
            offset: self.data.len() as u32,
            size,
        });
        self.data.reserve(size);
        unsafe {
            let target_ptr = self.data.as_mut_ptr().add(self.data.len());
            std::ptr::copy_nonoverlapping(&item as *const T as *const u8, target_ptr, size);
            self.data.set_len(self.data.len() + size);
        }
    }
    fn push_bool(&mut self, constant_id: u32, item: bool) {
        let size = std::mem::size_of::<vk::Bool32>();
        self.entries.push(vk::SpecializationMapEntry {
            constant_id,
            offset: self.data.len() as u32,
            size,
        });
        self.data.reserve(size);
        unsafe {
            let item: vk::Bool32 = if item { vk::TRUE } else { vk::FALSE };
            let target_ptr = self.data.as_mut_ptr().add(self.data.len());
            std::ptr::copy_nonoverlapping(
                &item as *const vk::Bool32 as *const u8,
                target_ptr,
                size,
            );
            self.data.set_len(self.data.len() + size);
        }
    }
}

#[derive(Clone, Debug)]
pub struct SpecializedShader<'a, S: Deref<Target = ShaderModule>> {
    pub stage: vk::ShaderStageFlags,
    pub flags: vk::PipelineShaderStageCreateFlags,
    pub shader: S,
    pub specialization_info: SpecializationInfo,
    pub entry_point: &'a CStr,
}
impl<'a, S: Deref<Target = ShaderModule>> HasDevice for SpecializedShader<'a, S> {
    fn device(&self) -> &Arc<Device> {
        &self.shader.device
    }
}
