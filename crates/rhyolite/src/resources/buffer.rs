use std::{
    ops::{Deref, DerefMut},
    pin::Pin,
    sync::Arc,
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
    Allocator, HasDevice, PhysicalDeviceMemoryModel, SharingMode, StagingRingBuffer,
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

/// Vocabulary:
/// asset: large buffers
/// static: buffers that never changes once initialized
impl Allocator {
    pub fn create_resident_buffer(
        &self,
        buffer_info: &vk::BufferCreateInfo,
        create_info: &vma::AllocationCreateInfo,
        alignment: u32,
    ) -> VkResult<ResidentBuffer> {
        let staging_buffer = if alignment == 0 {
            unsafe { self.inner().create_buffer(buffer_info, create_info)? }
        } else {
            unsafe {
                self.inner().create_buffer_with_alignment(
                    buffer_info,
                    create_info,
                    alignment as u64,
                )?
            }
        };
        Ok(ResidentBuffer {
            allocator: self.clone(),
            buffer: staging_buffer.0,
            allocation: staging_buffer.1,
            size: buffer_info.size,
        })
    }

    /// Create large uninitialized buffer only visible to the GPU.
    /// Discrete, Bar, ReBar, Unified: video memory.
    /// BiasedUnified: system memory.
    pub fn create_device_buffer_uninit(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        alignment: u32,
    ) -> VkResult<ResidentBuffer> {
        let buffer_create_info = vk::BufferCreateInfo {
            size,
            usage,
            ..Default::default()
        };
        let alloc_info = vma::AllocationCreateInfo {
            usage: match self.device().physical_device().memory_model() {
                PhysicalDeviceMemoryModel::BiasedUnified => vma::MemoryUsage::AutoPreferHost,
                _ => vma::MemoryUsage::AutoPreferDevice,
            },
            ..Default::default()
        };
        self.create_resident_buffer(&buffer_create_info, &alloc_info, alignment)
    }

    /// Create large uninitialized buffer accessible on both CPU and GPU.
    pub fn create_dynamic_buffer_uninit(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        alignment: u32,
    ) -> VkResult<ResidentBuffer> {
        let buffer_create_info = vk::BufferCreateInfo {
            size,
            usage,
            ..Default::default()
        };
        let alloc_info = vma::AllocationCreateInfo {
            usage: match self.device().physical_device().memory_model() {
                PhysicalDeviceMemoryModel::BiasedUnified => vma::MemoryUsage::AutoPreferHost,
                PhysicalDeviceMemoryModel::ReBar
                | PhysicalDeviceMemoryModel::Bar
                | PhysicalDeviceMemoryModel::Unified => vma::MemoryUsage::AutoPreferDevice,
                PhysicalDeviceMemoryModel::Discrete => {
                    panic!("Discrete GPUs do not have device-local, host-visible memory")
                }
            },
            flags: vma::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                | vma::AllocationCreateFlags::MAPPED,
            ..Default::default()
        };
        self.create_resident_buffer(&buffer_create_info, &alloc_info, alignment)
    }

    /// Create large initialized buffer only visible to the GPU.
    /// Discrete, Bar, ReBar: video memory, copy with staging buffer
    /// Unified: device-local, host-visible memory, direct write
    /// BiasedUnified: system memory, direct write
    pub fn create_static_device_buffer_with_data(
        &self,
        data: &[u8],
        usage: vk::BufferUsageFlags,
        alignment: u32,
        ring_buffer: &Arc<StagingRingBuffer>,
    ) -> VkResult<impl GPUCommandFuture<Output = RenderRes<ResidentBuffer>>> {
        let (dst_buffer, requires_staging_copy) =
            match self.device().physical_device().memory_model() {
                PhysicalDeviceMemoryModel::Discrete
                | PhysicalDeviceMemoryModel::Bar
                | PhysicalDeviceMemoryModel::ReBar => {
                    // Use a staging buffer for copying the initial data.
                    // Discrete: must use staging.
                    // Bar: 256MB of host visible memory is not enough for large buffers. Use staging
                    // ReBar: Can also use direct write, but the initialization only happens once,
                    // making it worthwhile to use a copy command. Memory mapping uses additional host-side address space.

                    // dst_buffer is device-local only.
                    let dst_buffer = self.create_device_buffer_uninit(
                        data.len() as u64,
                        usage | vk::BufferUsageFlags::TRANSFER_DST,
                        alignment,
                    )?;
                    (dst_buffer, true)
                }
                PhysicalDeviceMemoryModel::Unified | PhysicalDeviceMemoryModel::BiasedUnified => {
                    // Create upload buffer
                    // Unified: upload buffer is device-local, host-visible. direct write.
                    // BiasedUnified: upload buffer is system ram. direct write.
                    let dst_buffer =
                        self.create_dynamic_buffer_uninit(data.len() as u64, usage, alignment)?;
                    dst_buffer.contents_mut().unwrap()[0..data.len()].copy_from_slice(data);
                    (dst_buffer, false)
                }
            };
        let staging_buffer = if requires_staging_copy {
            let buffer = if data.len() > 1024 * 1024 {
                // For anything above 1MB, use a dedicated allocation. Heuristic.
                let staging_buffer = self.create_staging_buffer(data.len() as u64)?;
                staging_buffer.contents_mut().unwrap()[0..data.len()].copy_from_slice(data);
                Either::Left(staging_buffer)
            } else {
                Either::Right(ring_buffer.stage_changes(data))
            };
            Some(buffer)
        } else {
            None
        };
        Ok(commands! { move
            let mut dst_buffer = RenderRes::new(dst_buffer);
            if let Some(staging_buffer) = staging_buffer {
                let staging_buffer = RenderRes::new(staging_buffer);
                copy_buffer(&staging_buffer, &mut dst_buffer).await;
                retain!(staging_buffer);
            }
            let dst_buffer = dst_buffer;
            dst_buffer
        })
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
            0,
        )
    }

    // TODO: create download buffer
}
