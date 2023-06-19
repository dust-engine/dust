use std::{
    ops::{Deref, DerefMut},
    pin::Pin,
    task::Poll,
};

use ash::vk;
use ash::{prelude::VkResult, vk::Handle};
use pin_project::pin_project;

use crate::{
    debug::DebugObject,
    future::{CommandBufferRecordContext, GPUCommandFuture, RenderData, RenderRes, StageContext},
    macros::commands,
    utils::either::Either,
    Allocator, HasDevice, PhysicalDeviceMemoryModel, SharingMode,
};
use vma::Alloc;

pub trait BufferLike {
    fn raw_buffer(&self) -> vk::Buffer;
    fn offset(&self) -> vk::DeviceSize {
        0
    }
    fn size(&self) -> vk::DeviceSize;
    fn device_address(&self) -> vk::DeviceAddress;
    /// If the buffer is host visible and mapped, this function returns the host-side address.
    fn as_mut_ptr(&mut self) -> Option<*mut u8>;
}

impl BufferLike for vk::Buffer {
    fn raw_buffer(&self) -> vk::Buffer {
        *self
    }
    fn size(&self) -> vk::DeviceSize {
        vk::WHOLE_SIZE
    }
    fn device_address(&self) -> vk::DeviceAddress {
        panic!()
    }
    fn as_mut_ptr(&mut self) -> Option<*mut u8> {
        panic!()
    }
}

impl<A: BufferLike, B: BufferLike> BufferLike for Either<A, B> {
    fn raw_buffer(&self) -> vk::Buffer {
        match self {
            Either::Left(a) => a.raw_buffer(),
            Either::Right(a) => a.raw_buffer(),
        }
    }

    fn size(&self) -> vk::DeviceSize {
        match self {
            Either::Left(a) => a.size(),
            Either::Right(a) => a.size(),
        }
    }
    fn offset(&self) -> vk::DeviceSize {
        match self {
            Either::Left(a) => a.offset(),
            Either::Right(a) => a.offset(),
        }
    }

    fn device_address(&self) -> vk::DeviceAddress {
        match self {
            Either::Left(a) => a.device_address(),
            Either::Right(a) => a.device_address(),
        }
    }
    fn as_mut_ptr(&mut self) -> Option<*mut u8> {
        match self {
            Either::Left(a) => a.as_mut_ptr(),
            Either::Right(a) => a.as_mut_ptr(),
        }
    }
}

impl<T: BufferLike + ?Sized> BufferLike for Box<T> {
    fn raw_buffer(&self) -> vk::Buffer {
        (**self).raw_buffer()
    }

    fn size(&self) -> vk::DeviceSize {
        (**self).size()
    }
    fn offset(&self) -> vk::DeviceSize {
        (**self).offset()
    }

    fn device_address(&self) -> vk::DeviceAddress {
        (**self).device_address()
    }
    fn as_mut_ptr(&mut self) -> Option<*mut u8> {
        (**self).as_mut_ptr()
    }
}

pub struct BufferSlice<T: BufferLike> {
    buffer: T,
    offset: u64,
    size: u64,
}
impl<T: BufferLike + RenderData> RenderData for BufferSlice<T> {
    fn tracking_feedback(&mut self, feedback: &crate::future::TrackingFeedback) {
        self.buffer.tracking_feedback(feedback);
    }
}
impl<T: BufferLike> BufferLike for BufferSlice<T> {
    fn raw_buffer(&self) -> vk::Buffer {
        self.buffer.raw_buffer()
    }

    fn size(&self) -> vk::DeviceSize {
        self.size
    }

    fn device_address(&self) -> vk::DeviceAddress {
        self.buffer.device_address()
    }
    fn offset(&self) -> vk::DeviceSize {
        self.offset
    }
    fn as_mut_ptr(&mut self) -> Option<*mut u8> {
        self.buffer
            .as_mut_ptr()
            .map(|a| unsafe { a.add(self.offset as usize) })
    }
}

pub trait BufferExt: BufferLike {
    fn slice(self, offset: u64, size: u64) -> BufferSlice<Self>
    where
        Self: Sized,
    {
        assert!(offset + size <= self.size());
        let offset = self.offset() + offset;
        BufferSlice {
            buffer: self,
            offset,
            size,
        }
    }

    fn as_descriptor(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo {
            buffer: self.raw_buffer(),
            offset: self.offset(),
            range: self.size(),
        }
    }
}
impl<T: BufferLike> BufferExt for T {}
// Everyone wants a mutable refence to outer.
// Some people wants a mutable reference to inner.
// In the case of Fork. Each fork gets a & of the container. Container must be generic over &mut, and BorrowMut.
// Inner product must be generic over &mut and RefCell as well.

#[pin_project]
pub struct CopyBufferFuture<
    S: BufferLike + RenderData,
    T: BufferLike + RenderData,
    SRef: Deref<Target = RenderRes<S>>,
    TRef: DerefMut<Target = RenderRes<T>>,
> {
    pub src: SRef,
    pub dst: TRef,

    /// Buffer regions to copy. If empty, no-op. If None, copy everything.
    pub regions: Option<Vec<vk::BufferCopy>>,
}
impl<
        S: BufferLike + RenderData,
        T: BufferLike + RenderData,
        SRef: Deref<Target = RenderRes<S>>,
        TRef: DerefMut<Target = RenderRes<T>>,
    > GPUCommandFuture for CopyBufferFuture<S, T, SRef, TRef>
{
    type Output = ();
    type RetainedState = ();
    type RecycledState = ();
    #[inline]
    fn record(
        self: Pin<&mut Self>,
        ctx: &mut CommandBufferRecordContext,
        _recycled_state: &mut Self::RecycledState,
    ) -> Poll<(Self::Output, Self::RetainedState)> {
        let this = self.project();
        if let Some(regions) = this.regions && regions.is_empty() {
            return Poll::Ready(((), ()))
        }
        let src = this.src.deref().inner();
        let dst = this.dst.deref_mut().inner_mut();
        let entire_region = [vk::BufferCopy {
            src_offset: src.offset(),
            dst_offset: dst.offset(),
            size: src.size().min(dst.size()),
        }];
        ctx.record(|ctx, command_buffer| unsafe {
            ctx.device().cmd_copy_buffer(
                command_buffer,
                src.raw_buffer(),
                dst.raw_buffer(),
                if let Some(regions) = this.regions.as_ref() {
                    regions.as_slice()
                } else {
                    &entire_region
                },
            );
        });
        Poll::Ready(((), ()))
    }
    fn context(self: Pin<&mut Self>, ctx: &mut StageContext) {
        let this = self.project();
        ctx.read(
            this.src,
            vk::PipelineStageFlags2::COPY,
            vk::AccessFlags2::TRANSFER_READ,
        );

        ctx.write(
            this.dst,
            vk::PipelineStageFlags2::COPY,
            vk::AccessFlags2::TRANSFER_WRITE,
        );
    }
}

pub fn copy_buffer<
    S: BufferLike + RenderData,
    T: BufferLike + RenderData,
    SRef: Deref<Target = RenderRes<S>>,
    TRef: DerefMut<Target = RenderRes<T>>,
>(
    src: SRef,
    dst: TRef,
) -> CopyBufferFuture<S, T, SRef, TRef> {
    CopyBufferFuture {
        src,
        dst,
        regions: None,
    }
}
pub fn copy_buffer_regions<
    S: BufferLike + RenderData,
    T: BufferLike + RenderData,
    SRef: Deref<Target = RenderRes<S>>,
    TRef: DerefMut<Target = RenderRes<T>>,
>(
    src: SRef,
    dst: TRef,
    regions: Vec<vk::BufferCopy>,
) -> CopyBufferFuture<S, T, SRef, TRef> {
    CopyBufferFuture {
        src,
        dst,
        regions: Some(regions),
    }
}

#[pin_project]
pub struct UpdateBufferFuture<
    T: BufferLike + RenderData,
    TRef: DerefMut<Target = RenderRes<T>>,
    const N: usize,
> {
    pub dst: TRef,
    data: [u8; N],
}
impl<T: BufferLike + RenderData, TRef: DerefMut<Target = RenderRes<T>>, const N: usize>
    GPUCommandFuture for UpdateBufferFuture<T, TRef, N>
{
    type Output = ();
    type RetainedState = ();
    type RecycledState = ();
    #[inline]
    fn record(
        self: Pin<&mut Self>,
        ctx: &mut CommandBufferRecordContext,
        _recycled_state: &mut Self::RecycledState,
    ) -> Poll<(Self::Output, Self::RetainedState)> {
        let this = self.project();
        if this.data.is_empty() {
            return Poll::Ready(((), ()));
        }
        let offset = this.dst.inner.offset();
        let size = this.dst.inner.size().min(this.data.len() as u64) as usize;
        let dst = this.dst.deref_mut().inner_mut();
        let data: &[u8] = &this.data[..size];
        ctx.record(|ctx, command_buffer| unsafe {
            ctx.device()
                .cmd_update_buffer(command_buffer, dst.raw_buffer(), offset, data);
        });
        Poll::Ready(((), ()))
    }
    fn context(self: Pin<&mut Self>, ctx: &mut StageContext) {
        let this = self.project();

        ctx.write(
            this.dst,
            vk::PipelineStageFlags2::COPY,
            vk::AccessFlags2::TRANSFER_WRITE,
        );
    }
}

pub fn update_buffer<
    T: BufferLike + RenderData,
    TRef: DerefMut<Target = RenderRes<T>>,
    const N: usize,
>(
    dst: TRef,
    data: [u8; N],
) -> UpdateBufferFuture<T, TRef, N> {
    UpdateBufferFuture { dst, data }
}

#[pin_project]
pub struct FillBufferFuture<T: BufferLike + RenderData, TRef: DerefMut<Target = RenderRes<T>>> {
    pub dst: TRef,
    data: u32,
}
impl<T: BufferLike + RenderData, TRef: DerefMut<Target = RenderRes<T>>> GPUCommandFuture
    for FillBufferFuture<T, TRef>
{
    type Output = ();
    type RetainedState = ();
    type RecycledState = ();
    #[inline]
    fn record(
        self: Pin<&mut Self>,
        ctx: &mut CommandBufferRecordContext,
        _recycled_state: &mut Self::RecycledState,
    ) -> Poll<(Self::Output, Self::RetainedState)> {
        let this = self.project();
        let offset = this.dst.inner.offset();
        let size = this.dst.inner.size();
        let dst = this.dst.deref_mut().inner_mut();
        let data: u32 = *this.data;
        ctx.record(|ctx, command_buffer| unsafe {
            ctx.device()
                .cmd_fill_buffer(command_buffer, dst.raw_buffer(), offset, size, data);
        });
        Poll::Ready(((), ()))
    }
    fn context(self: Pin<&mut Self>, ctx: &mut StageContext) {
        let this = self.project();

        ctx.write(
            this.dst,
            vk::PipelineStageFlags2::COPY,
            vk::AccessFlags2::TRANSFER_WRITE,
        );
    }
}

pub fn fill_buffer<T: BufferLike + RenderData, TRef: DerefMut<Target = RenderRes<T>>>(
    dst: TRef,
    data: u32,
) -> FillBufferFuture<T, TRef> {
    FillBufferFuture { dst, data }
}

pub struct ResidentBuffer {
    allocator: Allocator,
    buffer: vk::Buffer,
    allocation: vma::Allocation,
    size: vk::DeviceSize,
}
impl RenderData for ResidentBuffer {}

impl ResidentBuffer {
    pub fn contents(&self) -> Option<&[u8]> {
        let info = self.allocator.inner().get_allocation_info(&self.allocation);
        if info.mapped_data.is_null() {
            None
        } else {
            unsafe {
                Some(std::slice::from_raw_parts(
                    info.mapped_data as *mut u8,
                    info.size as usize,
                ))
            }
        }
    }

    pub fn contents_mut(&self) -> Option<&mut [u8]> {
        let info = self.allocator.inner().get_allocation_info(&self.allocation);
        if info.mapped_data.is_null() {
            None
        } else {
            unsafe {
                Some(std::slice::from_raw_parts_mut(
                    info.mapped_data as *mut u8,
                    info.size as usize,
                ))
            }
        }
    }
}

impl BufferLike for ResidentBuffer {
    fn raw_buffer(&self) -> vk::Buffer {
        self.buffer
    }

    fn size(&self) -> vk::DeviceSize {
        self.size
    }

    fn device_address(&self) -> vk::DeviceAddress {
        unsafe {
            self.allocator
                .device()
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo {
                    buffer: self.buffer,
                    ..Default::default()
                })
        }
    }

    fn as_mut_ptr(&mut self) -> Option<*mut u8> {
        let info = self.allocator.inner().get_allocation_info(&self.allocation);
        if info.mapped_data.is_null() {
            None
        } else {
            Some(info.mapped_data as *mut u8)
        }
    }
}

impl HasDevice for ResidentBuffer {
    fn device(&self) -> &std::sync::Arc<crate::Device> {
        self.allocator.device()
    }
}

impl DebugObject for ResidentBuffer {
    fn object_handle(&mut self) -> u64 {
        self.buffer.as_raw()
    }

    const OBJECT_TYPE: vk::ObjectType = vk::ObjectType::BUFFER;
}

impl Drop for ResidentBuffer {
    fn drop(&mut self) {
        unsafe {
            self.allocator
                .inner()
                .destroy_buffer(self.buffer, &mut self.allocation);
        }
    }
}

#[derive(Default)]
pub struct BufferCreateInfo<'a> {
    pub flags: vk::BufferCreateFlags,
    pub size: vk::DeviceSize,
    pub usage: vk::BufferUsageFlags,
    pub sharing_mode: SharingMode<'a>,
}

impl Allocator {
    pub fn create_resident_buffer(
        &self,
        buffer_info: &vk::BufferCreateInfo,
        create_info: &vma::AllocationCreateInfo,
    ) -> VkResult<ResidentBuffer> {
        let staging_buffer = unsafe { self.inner().create_buffer(buffer_info, create_info)? };
        Ok(ResidentBuffer {
            allocator: self.clone(),
            buffer: staging_buffer.0,
            allocation: staging_buffer.1,
            size: buffer_info.size,
        })
    }
    pub fn create_resident_buffer_aligned(
        &self,
        buffer_info: &vk::BufferCreateInfo,
        create_info: &vma::AllocationCreateInfo,
        alignment: u32,
    ) -> VkResult<ResidentBuffer> {
        let staging_buffer = unsafe {
            self.inner()
                .create_buffer_with_alignment(buffer_info, create_info, alignment as u64)?
        };
        Ok(ResidentBuffer {
            allocator: self.clone(),
            buffer: staging_buffer.0,
            allocation: staging_buffer.1,
            size: buffer_info.size,
        })
    }
    /// Create uninitialized buffer visible to the CPU and local to the GPU.
    /// Only applicable to Bar, ReBar, Integrated memory architecture.
    pub fn create_write_buffer_uninit(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> VkResult<ResidentBuffer> {
        let buffer_create_info = vk::BufferCreateInfo {
            size,
            usage,
            ..Default::default()
        };
        let alloc_info = vma::AllocationCreateInfo {
            usage: vma::MemoryUsage::AutoPreferDevice,
            flags: vma::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                | vma::AllocationCreateFlags::MAPPED,
            required_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL
                | vk::MemoryPropertyFlags::HOST_VISIBLE,
            ..Default::default()
        };
        self.create_resident_buffer(&buffer_create_info, &alloc_info)
    }
    /// Create uninitialized buffer visible to the CPU and local to the GPU.
    /// Only applicable to Bar, ReBar, Integrated memory architecture.
    pub fn create_write_buffer_uninit_aligned(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        min_alignment: u64,
    ) -> VkResult<ResidentBuffer> {
        let buffer_create_info = vk::BufferCreateInfo {
            size,
            usage,
            ..Default::default()
        };
        let alloc_info = vma::AllocationCreateInfo {
            usage: vma::MemoryUsage::AutoPreferDevice,
            flags: vma::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                | vma::AllocationCreateFlags::MAPPED,
            required_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL
                | vk::MemoryPropertyFlags::HOST_VISIBLE,
            ..Default::default()
        };

        let (buffer, allocation) = unsafe {
            self.inner().create_buffer_with_alignment(
                &buffer_create_info,
                &alloc_info,
                min_alignment,
            )
        }?;
        Ok(ResidentBuffer {
            allocator: self.clone(),
            buffer,
            allocation,
            size,
        })
    }
    /// Create uninitialized buffer only visible to the GPU.
    pub fn create_device_buffer_uninit(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> VkResult<ResidentBuffer> {
        let buffer_create_info = vk::BufferCreateInfo {
            size,
            usage,
            ..Default::default()
        };
        let alloc_info = vma::AllocationCreateInfo {
            usage: vma::MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };
        self.create_resident_buffer(&buffer_create_info, &alloc_info)
    }
    /// Create uninitialized buffer only visible to the GPU.
    pub fn create_device_buffer_uninit_aligned(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        min_alignment: vk::DeviceSize,
    ) -> VkResult<ResidentBuffer> {
        let buffer_create_info = vk::BufferCreateInfo {
            size,
            usage,
            ..Default::default()
        };
        let alloc_info = vma::AllocationCreateInfo {
            usage: vma::MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };
        let (buffer, allocation) = unsafe {
            self.inner().create_buffer_with_alignment(
                &buffer_create_info,
                &alloc_info,
                min_alignment,
            )
        }?;
        Ok(ResidentBuffer {
            allocator: self.clone(),
            buffer,
            allocation,
            size,
        })
    }
    /// Crate a small host-visible buffer with uninitialized data, preferably local to the GPU.
    ///
    /// The buffer will be device-local, host visible on ResizableBar, Bar, and UMA memory models.
    /// The specified usage flags will be applied.
    ///
    /// The buffer will be host-visible on Discrete memory model. TRANSFER_SRC will be applied.
    ///
    /// Suitable for small amount of dynamic data, with device-local buffer created separately
    /// and transfers manually scheduled on devices without device-local, host-visible memory.
    pub fn create_dynamic_buffer_uninit(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> VkResult<ResidentBuffer> {
        let mut create_info = vk::BufferCreateInfo {
            size,
            usage,
            ..Default::default()
        };

        let dst_buffer = match self.device().physical_device().memory_model() {
            PhysicalDeviceMemoryModel::UMA
            | PhysicalDeviceMemoryModel::Bar
            | PhysicalDeviceMemoryModel::ResizableBar => {
                let buf = self.create_resident_buffer(
                    &create_info,
                    &vma::AllocationCreateInfo {
                        flags: vma::AllocationCreateFlags::MAPPED
                            | vma::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                        usage: vma::MemoryUsage::AutoPreferDevice,
                        required_flags: vk::MemoryPropertyFlags::empty(),
                        preferred_flags: vk::MemoryPropertyFlags::empty(),
                        memory_type_bits: 0,
                        user_data: 0,
                        priority: 0.0,
                    },
                )?;
                buf
            }
            PhysicalDeviceMemoryModel::Discrete => {
                create_info.usage |= vk::BufferUsageFlags::TRANSFER_SRC;
                let dst_buffer = self.create_resident_buffer(
                    &create_info,
                    &vma::AllocationCreateInfo {
                        flags: vma::AllocationCreateFlags::MAPPED
                            | vma::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                        usage: vma::MemoryUsage::AutoPreferHost,
                        ..Default::default()
                    },
                )?;
                dst_buffer
            }
        };
        Ok(dst_buffer)
    }

    pub fn create_dynamic_buffer_uninit_aligned(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        alignment: u64,
    ) -> VkResult<ResidentBuffer> {
        let mut create_info = vk::BufferCreateInfo {
            size,
            usage,
            ..Default::default()
        };

        let dst_buffer = match self.device().physical_device().memory_model() {
            PhysicalDeviceMemoryModel::UMA
            | PhysicalDeviceMemoryModel::Bar
            | PhysicalDeviceMemoryModel::ResizableBar => unsafe {
                let (buf, alloc) = self.inner().create_buffer_with_alignment(
                    &create_info,
                    &vma::AllocationCreateInfo {
                        flags: vma::AllocationCreateFlags::MAPPED
                            | vma::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                        usage: vma::MemoryUsage::AutoPreferDevice,
                        required_flags: vk::MemoryPropertyFlags::empty(),
                        preferred_flags: vk::MemoryPropertyFlags::empty(),
                        memory_type_bits: 0,
                        user_data: 0,
                        priority: 0.0,
                    },
                    alignment,
                )?;
                ResidentBuffer {
                    allocator: self.clone(),
                    buffer: buf,
                    allocation: alloc,
                    size,
                }
            },
            PhysicalDeviceMemoryModel::Discrete => unsafe {
                create_info.usage |= vk::BufferUsageFlags::TRANSFER_SRC;
                let (buffer, allocation) = self.inner().create_buffer_with_alignment(
                    &create_info,
                    &vma::AllocationCreateInfo {
                        flags: vma::AllocationCreateFlags::MAPPED
                            | vma::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                        usage: vma::MemoryUsage::AutoPreferHost,
                        ..Default::default()
                    },
                    alignment,
                )?;

                ResidentBuffer {
                    allocator: self.clone(),
                    buffer,
                    allocation,
                    size,
                }
            },
        };
        Ok(dst_buffer)
    }

    /// Crate a small device-local buffer with uninitialized data, guaranteed local to the GPU.
    /// The data will be host visible on ResizableBar, Bar, and UMA memory models.
    /// TRANSFER_DST usage flag will be automatically added to the created buffer.
    /// Suitable for small amount of dynamic data, with staging buffer created separately.
    /// TODO: rename to create_dynamic_upload_buffer_uninit
    pub fn create_upload_buffer_uninit(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        alignment: u32,
    ) -> VkResult<ResidentBuffer> {
        let mut create_info = vk::BufferCreateInfo {
            size,
            usage,
            ..Default::default()
        };

        let dst_buffer = match self.device().physical_device().memory_model() {
            PhysicalDeviceMemoryModel::UMA
            | PhysicalDeviceMemoryModel::Bar
            | PhysicalDeviceMemoryModel::ResizableBar => {
                let buf = self.create_resident_buffer_aligned(
                    &create_info,
                    &vma::AllocationCreateInfo {
                        flags: vma::AllocationCreateFlags::MAPPED
                            | vma::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                        usage: vma::MemoryUsage::AutoPreferDevice,
                        required_flags: vk::MemoryPropertyFlags::empty(),
                        preferred_flags: vk::MemoryPropertyFlags::empty(),
                        memory_type_bits: 0,
                        user_data: 0,
                        priority: 0.0,
                    },
                    alignment,
                )?;
                buf
            }
            PhysicalDeviceMemoryModel::Discrete => {
                create_info.usage |= vk::BufferUsageFlags::TRANSFER_DST;
                let dst_buffer = self.create_resident_buffer_aligned(
                    &create_info,
                    &vma::AllocationCreateInfo {
                        flags: vma::AllocationCreateFlags::empty(),
                        usage: vma::MemoryUsage::AutoPreferDevice,
                        ..Default::default()
                    },
                    alignment,
                )?;
                dst_buffer
            }
        };
        Ok(dst_buffer)
    }

    /// Crate a large device-local buffer with uninitialized data, guaranteed local to the GPU.
    /// Suitable for large assets with occasionally updated regions.
    ///
    /// The data will be host visible on ResizableBar and UMA memory models. On these memory
    /// architectures, the application may update the buffer directly when it's not already in use
    /// by the GPU.
    ///
    /// The data will not be host visible on Bar and Discrete memory models. The application must
    /// use a staging buffer for the updates. TRANSFER_DST usage flag will be automatically added
    /// to the created buffer.
    pub fn create_dynamic_asset_buffer_uninit(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> VkResult<ResidentBuffer> {
        let mut create_info = vk::BufferCreateInfo {
            size,
            usage,
            ..Default::default()
        };

        let dst_buffer = match self.device().physical_device().memory_model() {
            PhysicalDeviceMemoryModel::UMA | PhysicalDeviceMemoryModel::ResizableBar => {
                let buf = self.create_resident_buffer(
                    &create_info,
                    &vma::AllocationCreateInfo {
                        flags: vma::AllocationCreateFlags::MAPPED
                            | vma::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                        usage: vma::MemoryUsage::AutoPreferDevice,
                        required_flags: vk::MemoryPropertyFlags::empty(),
                        preferred_flags: vk::MemoryPropertyFlags::empty(),
                        memory_type_bits: 0,
                        user_data: 0,
                        priority: 0.0,
                    },
                )?;
                buf
            }
            PhysicalDeviceMemoryModel::Discrete | PhysicalDeviceMemoryModel::Bar => {
                create_info.usage |= vk::BufferUsageFlags::TRANSFER_DST;
                let dst_buffer = self.create_resident_buffer(
                    &create_info,
                    &vma::AllocationCreateInfo {
                        flags: vma::AllocationCreateFlags::empty(),
                        usage: vma::MemoryUsage::AutoPreferDevice,
                        ..Default::default()
                    },
                )?;
                dst_buffer
            }
        };
        Ok(dst_buffer)
    }

    pub fn create_staging_buffer(&self, size: vk::DeviceSize) -> VkResult<ResidentBuffer> {
        self.create_resident_buffer(
            &vk::BufferCreateInfo {
                size,
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                ..Default::default()
            },
            &vma::AllocationCreateInfo {
                flags: vma::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                    | vma::AllocationCreateFlags::MAPPED,
                usage: vma::MemoryUsage::AutoPreferHost,
                ..Default::default()
            },
        )
    }

    /// Crate a small device-local buffer with a writer callback, only visible to the GPU.
    /// The data will be directly written to the buffer on ResizableBar, Bar, and UMA memory models.
    /// We will create a temporary staging buffer on Discrete GPUs with no host-accessible device-local memory.
    /// TRANSFER_DST usage flag will be automatically added to the created buffer.
    /// Suitable for small amount of dynamic data.
    /// TODO: rename to create_dynamic_buffer_with_writer
    pub fn create_device_buffer_with_writer(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        writer: impl for<'a> FnOnce(&'a mut [u8]),
    ) -> VkResult<impl GPUCommandFuture<Output = RenderRes<ResidentBuffer>>> {
        let dst_buffer = self.create_upload_buffer_uninit(size, usage, 0)?;
        let staging_buffer = if let Some(contents) = dst_buffer.contents_mut() {
            writer(contents);
            None
        } else {
            let staging_buffer = self.create_staging_buffer(size)?;
            writer(staging_buffer.contents_mut().unwrap());
            Some(staging_buffer)
        };

        Ok(commands! {
            let mut dst_buffer = RenderRes::new(dst_buffer);
            if let Some(staging_buffer) = staging_buffer {
                let staging_buffer = RenderRes::new(staging_buffer);
                copy_buffer(&staging_buffer, &mut dst_buffer).await;
                retain!(staging_buffer);
            }
            dst_buffer
        })
    }

    /// Crate a small device-local buffer with pre-populated data, only visible to the GPU.
    /// The data will be directly written to the buffer on ResizableBar, Bar, and UMA memory models.
    /// We will create a temporary staging buffer on Discrete GPUs with no host-accessible device-local memory.
    /// TRANSFER_DST usage flag will be automatically added to the created buffer.
    /// Suitable for small amount of dynamic data.
    /// TODO: rename to create_dynamic_buffer_with_data
    pub fn create_device_buffer_with_data(
        &self,
        data: &[u8],
        usage: vk::BufferUsageFlags,
    ) -> VkResult<impl GPUCommandFuture<Output = RenderRes<ResidentBuffer>>> {
        let dst_buffer = self.create_upload_buffer_uninit(data.len() as u64, usage, 0)?;
        let staging_buffer = if let Some(contents) = dst_buffer.contents_mut() {
            contents[..data.len()].copy_from_slice(data);
            None
        } else {
            let staging_buffer = self.create_staging_buffer(data.len() as u64)?;
            staging_buffer.contents_mut().unwrap()[..data.len()].copy_from_slice(data);
            Some(staging_buffer)
        };

        Ok(commands! {
            let mut dst_buffer = RenderRes::new(dst_buffer);
            if let Some(staging_buffer) = staging_buffer {
                let staging_buffer = RenderRes::new(staging_buffer);
                copy_buffer(&staging_buffer, &mut dst_buffer).await;
                retain!(staging_buffer);
            }
            dst_buffer
        })
    }

    /// Crate a large device-local buffer with a writer callback, only visible to the GPU.
    /// The data will be directly written to the buffer on ResizableBar, and UMA memory models.
    /// We will create a temporary staging buffer on Bar and Discrete GPUs with no host-accessible device-local memory.
    /// TRANSFER_DST usage flag will be automatically added to the created buffer.
    /// Suitable for large assets with some regions updated occasionally.
    pub fn create_dynamic_asset_buffer_with_writer(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        writer: impl for<'a> FnOnce(&'a mut [u8]),
    ) -> VkResult<impl GPUCommandFuture<Output = RenderRes<ResidentBuffer>>> {
        let dst_buffer = self.create_upload_buffer_uninit(size, usage, 0)?;
        let staging_buffer = if let Some(contents) = dst_buffer.contents_mut() {
            writer(contents);
            None
        } else {
            let staging_buffer = self.create_staging_buffer(size)?;
            writer(staging_buffer.contents_mut().unwrap());
            Some(staging_buffer)
        };

        Ok(commands! {
            let mut dst_buffer = RenderRes::new(dst_buffer);
            if let Some(staging_buffer) = staging_buffer {
                let staging_buffer = RenderRes::new(staging_buffer);
                copy_buffer(&staging_buffer, &mut dst_buffer).await;
                retain!(staging_buffer);
            }
            dst_buffer
        })
    }

    /// Crate a large device-local buffer with a writer callback, only visible to the GPU.
    /// The data will be directly written to the buffer on ResizableBar, and UMA memory models.
    /// We will create a temporary staging buffer on Bar and Discrete GPUs with no host-accessible device-local memory.
    /// TRANSFER_DST usage flag will be automatically added to the created buffer.
    /// Suitable for large assets with some regions updated occasionally.
    pub fn create_dynamic_asset_buffer_with_data(
        &self,
        data: &[u8],
        usage: vk::BufferUsageFlags,
        alignment: u32,
    ) -> VkResult<impl GPUCommandFuture<Output = RenderRes<ResidentBuffer>>> {
        let dst_buffer = self.create_upload_buffer_uninit(data.len() as u64, usage, alignment)?;
        let staging_buffer = if let Some(contents) = dst_buffer.contents_mut() {
            contents[..data.len()].copy_from_slice(data);
            None
        } else {
            let staging_buffer = self.create_staging_buffer(data.len() as u64)?;
            staging_buffer.contents_mut().unwrap()[..data.len()].copy_from_slice(data);
            Some(staging_buffer)
        };

        Ok(commands! {
            let mut dst_buffer = RenderRes::new(dst_buffer);
            if let Some(staging_buffer) = staging_buffer {
                let staging_buffer = RenderRes::new(staging_buffer);
                copy_buffer(&staging_buffer, &mut dst_buffer).await;
                retain!(staging_buffer);
            }
            dst_buffer
        })
    }
}
