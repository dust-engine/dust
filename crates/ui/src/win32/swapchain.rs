//! The game's [`Swapchain`](crate::Swapchain) on Windows: buffers that the game
//! renders into and that the composition swapchain API (`IPresentationManager`)
//! presents to the `GameLayer`'s visual. Nothing is copied on the way.
//!
//! The buffers are D3D11 textures on Vulkan's GPU, imported into Vulkan
//! (`VK_KHR_external_memory_win32`) and registered with a presentation manager.
//! Each frame, [`acquire`] picks the next of three, waiting until the compositor is
//! done with it and making it again if the game's size or color space has
//! changed, and has the submission of [`SwapchainSet`] signal a fence shared
//! with D3D11 (`VK_KHR_external_semaphore_win32`); [`release`] hands the buffer
//! over to D3D11 (`VK_QUEUE_FAMILY_EXTERNAL`); and [`present`] presents it, once
//! that fence is signaled.

use std::sync::Arc;

use bevy::ecs::system::NonSendMarker;
use bevy::prelude::*;
use bevy_pumicite::{
    CreateDevice, DefaultRenderSet, PumiciteApp, SubmissionSetConfig, SubmissionState,
    pumicite::{
        Device,
        ash::{khr, vk, vk::TaggedStructure as _},
        device::DeviceBuilder,
        image::ImageLike,
        project_host_metadata,
        sync::GPUMutex,
        tracking::{Access, ResourceState},
        utils::AsVkHandle,
    },
    queue::{Queue, RenderQueue},
};
use windows::Win32::Foundation::{
    GENERIC_ALL, HANDLE, HMODULE, LUID, PRESENTATION_ERROR_LOST, RECT, WAIT_OBJECT_0,
};
use windows::Win32::Graphics::CompositionSwapchain::{
    CreatePresentationFactory, ICompositionFramePresentStatistics, IPresentStatusPresentStatistics,
    IPresentationBuffer, IPresentationFactory, IPresentationManager, IPresentationSurface,
    PresentStatisticsKind_CompositionFrame, PresentStatisticsKind_PresentStatus,
    PresentStatus_Skipped,
};
use windows::Win32::Graphics::Direct3D::{
    D3D_DRIVER_TYPE_UNKNOWN, D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_11_1,
};
use windows::Win32::Graphics::Direct3D11::{
    D3D11_BIND_RENDER_TARGET, D3D11_BIND_SHADER_RESOURCE, D3D11_BIND_UNORDERED_ACCESS,
    D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_CREATE_DEVICE_FLAG,
    D3D11_CREATE_DEVICE_PREVENT_INTERNAL_THREADING_OPTIMIZATIONS,
    D3D11_CREATE_DEVICE_SINGLETHREADED, D3D11_FEATURE_DATA_DISPLAYABLE, D3D11_FEATURE_DISPLAYABLE,
    D3D11_FENCE_FLAG_SHARED, D3D11_RESOURCE_MISC_SHARED, D3D11_RESOURCE_MISC_SHARED_DISPLAYABLE,
    D3D11_RESOURCE_MISC_SHARED_NTHANDLE, D3D11_SDK_VERSION, D3D11_TEXTURE2D_DESC,
    D3D11_USAGE_DEFAULT, D3D11CreateDevice, ID3D11Device, ID3D11Device5, ID3D11DeviceContext4,
    ID3D11Fence, ID3D11Texture2D,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_ALPHA_MODE_IGNORE, DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709,
    DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709, DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020,
    DXGI_COLOR_SPACE_TYPE, DXGI_FORMAT_R10G10B10A2_UNORM, DXGI_SAMPLE_DESC,
};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, DXGI_ERROR_UNSUPPORTED, IDXGIAdapter1,
    IDXGIFactory4, IDXGIResource1,
};
use windows::Win32::System::Threading::WaitForSingleObject;
use windows::core::{Interface, Owned, PCWSTR};

use crate::win32::layer::GameLayerSharedState;

/// How many buffers the game renders into in turn: one being written, one queued
/// for the compositor, and one on screen.
const BUFFER_COUNT: usize = 3;


/// Where systems using the [`Swapchain`]'s current image run: in a submission of
/// their own on the render queue, after the submission of `DefaultRenderSet`, and
/// between [`acquire`] and [`release`]. It's skipped on frames without an image.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SwapchainSet;

/// The submission set that [`acquire`], [`SwapchainSet`] and [`release`] are in.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SwapchainSubmissionSet;

/// The run condition for [`SwapchainSet`].
fn swapchain_image_acquired(swapchain: Option<Res<Swapchain>>) -> bool {
    swapchain.is_some_and(|swapchain| swapchain.current_image().is_some())
}

/// The composition surface handle the swapchain presents to. The `GameLayer`'s
/// visual shows the surface.
#[derive(Clone, Copy)]
pub(super) struct SurfaceHandle(pub(super) HANDLE);

// SAFETY: a handle names the same kernel object on every thread of the process.
// Only the Bevy thread presents to it.
unsafe impl Send for SurfaceHandle {}
unsafe impl Sync for SurfaceHandle {}

/// Inserts the [`Swapchain`] and creates its presenter at startup, puts
/// [`SwapchainSet`] in a submission that follows the submission of
/// `DefaultRenderSet`, and adds the systems that acquire, release and present the
/// buffers around the set.
pub(super) struct CompositionSwapchainPlugin {
    pub(super) surface: SurfaceHandle,
    pub(super) game_layer_state: Arc<GameLayerSharedState>,
}

impl Plugin for CompositionSwapchainPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(Swapchain::new(self.surface, self.game_layer_state.clone()))
            .add_systems(
                Startup,
                (|mut device_builder: ResMut<DeviceBuilder>| {
                    device_builder
                        .enable_extension::<khr::external_memory_win32::Meta>()
                        .unwrap();
                    device_builder
                        .enable_extension::<khr::external_semaphore_win32::Meta>()
                        .unwrap();
                })
                .before(CreateDevice),
            )
            .add_systems(Startup, create_presenter.after(CreateDevice))
            .add_submission_set::<RenderQueue>(
                SwapchainSubmissionSet,
                SubmissionSetConfig {
                    debug_color: Vec4::new(1.0, 0.0, 1.0, 1.0),
                },
            )
            .configure_sets(
                PostUpdate,
                (
                    SwapchainSet
                        .in_set(SwapchainSubmissionSet)
                        .run_if(swapchain_image_acquired),
                    DefaultRenderSet.before(SwapchainSubmissionSet),
                ),
            )
            .add_systems(
                PostUpdate,
                (
                    acquire.in_set(SwapchainSubmissionSet).before(SwapchainSet),
                    release.in_set(SwapchainSubmissionSet).after(SwapchainSet),
                    present.after(SwapchainSubmissionSet),
                ),
            );
    }
}

/// The swapchain's state on Windows: the presentation manager and the buffers the
/// game renders into.
///
/// It's inserted without a presenter, which [`create_presenter`] creates at
/// startup, once the Vulkan device exists. It's removed if the system doesn't
/// support the composition swapchain API, or if the presentation manager is lost
/// and another can't be created; the systems that use it are skipped then.
#[derive(Resource)]
pub struct Swapchain {
    surface: SurfaceHandle,
    /// The game's size in physical pixels, which the `GameLayer` sets.
    game_layer_state: Arc<GameLayerSharedState>,
    /// The presentation manager and what goes with it. [`present`] replaces it
    /// whenever the presentation manager is lost. It's `None` until
    /// [`create_presenter`] creates it, and while [`present`] replaces it: the
    /// lost presentation manager has to let go of the surface before another can
    /// present to it, so it's dropped first.
    presenter: Option<PresentationManager>,
    /// The buffers, used in turn. Each is made when it's first used, and made again
    /// when it's next used after the game's size or color space changes.
    buffers: [Option<Buffer>; BUFFER_COUNT],
    /// The index in `buffers` of the current frame's buffer, or of the last
    /// frame's between frames.
    index: usize,
    /// Set from [`acquire`] until [`present`], while `buffers[index]` is the
    /// current frame's buffer: the value of `render_done` its present waits for.
    current: Option<u64>,
    /// The current image's resource state.
    state: ResourceState,
}

// SAFETY: the D3D11 and presentation objects are only used by the systems taking
// `NonSendMarker`, which run on the Bevy thread; the app is dropped there too.
// `extent` reads an atomic. The rest is `Send` and `Sync`.
unsafe impl Send for Swapchain {}
unsafe impl Sync for Swapchain {}

impl Swapchain {
    fn new(surface: SurfaceHandle, game_layer_state: Arc<GameLayerSharedState>) -> Self {
        Swapchain {
            surface,
            game_layer_state,
            presenter: None,
            buffers: Default::default(),
            index: 0,
            current: None,
            state: ResourceState::default(),
        }
    }

    /// The size of the game's images, known for the whole frame, e.g. to size the
    /// images that `DefaultRenderSet` renders.
    pub fn extent(&self) -> UVec2 {
        self.game_layer_state.size()
    }

    /// The image the current frame renders into, from its acquisition until it's
    /// presented: in [`SwapchainSet`].
    pub fn current_image(&self) -> Option<&GPUMutex<PresentationImage>> {
        self.current
            .and(self.buffers[self.index].as_ref())
            .map(|buffer| &buffer.image)
    }

    /// The current image's resource state, for the barriers of the work using it.
    pub fn state_mut(&mut self) -> &mut ResourceState {
        &mut self.state
    }

    /// Replaces the lost presentation manager with another, and drops the buffers
    /// registered with it. If another can't be created, the swapchain is left
    /// without one.
    fn replace_lost_presenter(&mut self) -> windows::core::Result<()> {
        tracing::warn!("The presentation manager was lost; creating another");
        self.buffers = Default::default();
        // The lost presentation manager has to let go of the surface before
        // another can present to it.
        let lost = self.presenter.take().unwrap();
        let device = lost.device.clone();
        drop(lost);
        self.presenter = Some(PresentationManager::new(&device, self.surface)?);
        Ok(())
    }
}

/// A buffer the game renders into: a D3D11 texture registered with the
/// presentation manager, and imported into Vulkan.
///
/// It can be dropped at any time. Only `image` is used by Vulkan commands, and
/// the `GPUMutex` keeps it until the GPU is done with them. The presentation
/// manager keeps its own reference to the texture while the buffer is on
/// screen, and the imported memory keeps one too.
struct Buffer {
    image: GPUMutex<PresentationImage>,
    presentation: IPresentationBuffer,
    /// Set while the compositor is done with the buffer: from when the buffer's last
    /// present retires until the buffer is presented again.
    available: Owned<HANDLE>,
}

/// The presentation manager, the D3D11 device it presents on, and the fence that
/// orders work between D3D11 and Vulkan.
struct PresentationManager {
    device: Device,
    d3d: ID3D11Device5,
    context: ID3D11DeviceContext4,
    /// Whether the display hardware can scan the buffers out directly, without
    /// composition.
    displayable: bool,
    manager: IPresentationManager,
    surface: IPresentationSurface,
    /// Signaled by the submission of [`SwapchainSet`] once it's done writing a
    /// buffer; the buffer's present waits for it.
    render_done: ID3D11Fence,
    /// `render_done`, imported into Vulkan.
    render_done_semaphore: vk::Semaphore,
    render_done_value: u64,
    /// The color space that `surface` shows its buffer in. [`PresentationManager::present`]
    /// changes it to the color space of the buffer it presents.
    color_space: vk::ColorSpaceKHR,
    /// Set while the presentation manager has statistics for
    /// [`PresentationManager::read_statistics`] to read.
    statistics_available: Owned<HANDLE>,
    /// The color space that the game should present in: HDR10 while a display in
    /// HDR mode shows the game, and sRGB otherwise. It starts as sRGB, until
    /// [`PresentationManager::read_statistics`] finds otherwise.
    preferred_color_space: vk::ColorSpaceKHR,
    /// Signaled with the ID of each present as it starts retiring, which is when
    /// the present after it is queued for display.
    retiring: ID3D11Fence,
}

impl PresentationManager {
    /// Creates a presentation manager on Vulkan's GPU, presenting to `surface`.
    fn new(device: &Device, surface: SurfaceHandle) -> windows::core::Result<Self> {
        let id = device
            .physical_device()
            .properties()
            .get::<vk::PhysicalDeviceIDProperties>();
        assert_eq!(id.device_luid_valid, vk::TRUE, "the GPU has no LUID");
        let luid = LUID {
            LowPart: u32::from_le_bytes(id.device_luid[..4].try_into().unwrap()),
            HighPart: i32::from_le_bytes(id.device_luid[4..].try_into().unwrap()),
        };
        let factory: IDXGIFactory4 = unsafe { CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)) }?;
        let adapter: IDXGIAdapter1 = unsafe { factory.EnumAdapterByLuid(luid) }?;
        let mut d3d: Option<ID3D11Device> = None;
        let mut context = None;
        unsafe {
            D3D11CreateDevice(
                &adapter,
                D3D_DRIVER_TYPE_UNKNOWN,
                HMODULE::default(),
                D3D11_CREATE_DEVICE_FLAG(
                    D3D11_CREATE_DEVICE_BGRA_SUPPORT.0
                        | D3D11_CREATE_DEVICE_SINGLETHREADED.0
                        | D3D11_CREATE_DEVICE_PREVENT_INTERNAL_THREADING_OPTIMIZATIONS.0,
                ),
                Some(&[D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0]),
                D3D11_SDK_VERSION,
                Some(&mut d3d),
                None,
                Some(&mut context),
            )
        }?;
        let d3d = d3d.unwrap();
        let mut displayable = D3D11_FEATURE_DATA_DISPLAYABLE::default();
        let displayable = unsafe {
            d3d.CheckFeatureSupport(
                D3D11_FEATURE_DISPLAYABLE,
                (&raw mut displayable).cast(),
                size_of::<D3D11_FEATURE_DATA_DISPLAYABLE>() as u32,
            )
        }
        .is_ok_and(|()| displayable.DisplayableTexture.as_bool());

        let presentation: IPresentationFactory = unsafe { CreatePresentationFactory(&d3d) }?;
        if unsafe { presentation.IsPresentationSupported() } == 0 {
            return Err(windows::core::Error::new(
                DXGI_ERROR_UNSUPPORTED,
                "the composition swapchain API isn't supported on this system (it needs Windows 11 and a WDDM 2.0 driver)",
            ));
        }
        let manager = unsafe { presentation.CreatePresentationManager() }?;
        let surface = unsafe { manager.CreatePresentationSurface(surface.0) }?;
        let color_space = vk::ColorSpaceKHR::SRGB_NONLINEAR;
        unsafe {
            surface.SetAlphaMode(DXGI_ALPHA_MODE_IGNORE)?;
            surface.SetColorSpace(dxgi_color_space(color_space))?;
        }
        // Each present's composition-frame statistics name the outputs that
        // showed the present, and the color space of each output.
        unsafe { manager.EnablePresentStatisticsKind(PresentStatisticsKind_CompositionFrame, 1) }?;
        // Each present's status statistics say whether it was shown or skipped.
        unsafe { manager.EnablePresentStatisticsKind(PresentStatisticsKind_PresentStatus, 1) }?;
        let statistics_available =
            unsafe { Owned::new(manager.GetPresentStatisticsAvailableEvent()?) };
        let retiring = unsafe { manager.GetPresentRetiringFence::<ID3D11Fence>() }?;

        let d3d: ID3D11Device5 = d3d.cast()?;
        let mut render_done: Option<ID3D11Fence> = None;
        unsafe { d3d.CreateFence(0, D3D11_FENCE_FLAG_SHARED, &mut render_done) }?;
        let render_done = render_done.unwrap();
        let shared = unsafe {
            Owned::new(render_done.CreateSharedHandle(None, GENERIC_ALL.0, PCWSTR::null())?)
        };
        let mut timeline = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let render_done_semaphore = unsafe {
            device.create_semaphore(
                &vk::SemaphoreCreateInfo::default().push(&mut timeline),
                None,
            )
        }
        .unwrap();
        // Importing an NT handle doesn't take ownership of it, so `shared` still closes it.
        unsafe {
            device
                .extension::<khr::external_semaphore_win32::Meta>()
                .import_semaphore_win32_handle(
                    &vk::ImportSemaphoreWin32HandleInfoKHR::default()
                        .semaphore(render_done_semaphore)
                        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::D3D11_FENCE)
                        .handle(shared.0 as vk::HANDLE),
                )
                .unwrap();
        }

        Ok(PresentationManager {
            device: device.clone(),
            d3d,
            context: context.unwrap().cast()?,
            displayable,
            manager,
            surface,
            render_done,
            render_done_semaphore,
            render_done_value: 0,
            color_space,
            statistics_available,
            preferred_color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            retiring,
        })
    }

    /// Reads the statistics of the presents since the last call: logs the presents
    /// that were skipped, and updates `preferred_color_space`. The color space
    /// comes from the composition frame that first showed each present, so it
    /// follows the window from one display to another, and it follows HDR being
    /// turned on or off.
    fn read_statistics(&mut self) {
        // Reading every statistic also keeps the queue from overflowing.
        while (unsafe { WaitForSingleObject(*self.statistics_available, 0) }) == WAIT_OBJECT_0 {
            let Ok(statistics) = (unsafe { self.manager.GetNextPresentStatistics() }) else {
                break;
            };
            if let Ok(status) = statistics.cast::<IPresentStatusPresentStatistics>() {
                if unsafe { status.GetPresentStatus() } == PresentStatus_Skipped {
                    tracing::debug!("Present {} was skipped", unsafe {
                        statistics.GetPresentId()
                    });
                }
                continue;
            }
            let Ok(frame) = statistics.cast::<ICompositionFramePresentStatistics>() else {
                continue;
            };
            let mut count = 0;
            let mut instances = std::ptr::null_mut();
            unsafe { frame.GetDisplayInstanceArray(&mut count, &mut instances) };
            if instances.is_null() || count == 0 {
                continue;
            }
            // SAFETY: the statistics object owns the array, and outlives this use.
            let instances = unsafe { std::slice::from_raw_parts(instances, count as usize) };
            // There's one instance for each output that showed the present, and
            // its color space is the one that the compositor composes in for that
            // output. For an output in HDR mode that's linear scRGB
            // (`G10_NONE_P709`), not the HDR10 signal that the display receives.
            // For an output in SDR mode it's sRGB. A window that spans an HDR
            // display and an SDR display presents in HDR10, which the compositor
            // converts for the SDR display.
            let color_space = if instances.iter().any(|instance| {
                instance.colorSpace == DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709
                    || instance.colorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020
            }) {
                vk::ColorSpaceKHR::HDR10_ST2084_EXT
            } else {
                vk::ColorSpaceKHR::SRGB_NONLINEAR
            };
            if color_space != self.preferred_color_space {
                tracing::info!("The game now presents in {color_space:?}");
            }
            self.preferred_color_space = color_space;
        }
    }

    /// Waits until the compositor is done with `buffer`, for up to `timeout`. Only
    /// then can Vulkan write the buffer: the presentation API holds back D3D work
    /// on a buffer that's still in use, but not Vulkan work. Returns whether the
    /// buffer is available.
    fn wait_available(&self, buffer: &Buffer, timeout_ms: u32) -> bool {
        (unsafe { WaitForSingleObject(*buffer.available, timeout_ms) }) == WAIT_OBJECT_0
    }

    /// Creates a buffer of `size` for an image in `color_space`, registered with
    /// the presentation manager and imported into Vulkan.
    fn create_buffer(
        &self,
        size: UVec2,
        color_space: vk::ColorSpaceKHR,
    ) -> windows::core::Result<Buffer> {
        let mut misc = D3D11_RESOURCE_MISC_SHARED.0 | D3D11_RESOURCE_MISC_SHARED_NTHANDLE.0;
        if self.displayable {
            misc |= D3D11_RESOURCE_MISC_SHARED_DISPLAYABLE.0;
        }
        let desc = D3D11_TEXTURE2D_DESC {
            Width: size.x,
            Height: size.y,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_R10G10B10A2_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            // The presentation API requires RGB buffers to be bound as shader
            // resources. The unordered-access and render-target bindings allow
            // the Vulkan image's storage and color-attachment usages, and tonemap
            // writes the image as a storage image.
            BindFlags: (D3D11_BIND_SHADER_RESOURCE.0
                | D3D11_BIND_UNORDERED_ACCESS.0
                | D3D11_BIND_RENDER_TARGET.0) as u32,
            CPUAccessFlags: 0,
            MiscFlags: misc as u32,
        };
        let mut texture = None;
        unsafe { self.d3d.CreateTexture2D(&desc, None, Some(&mut texture)) }?;
        let texture: ID3D11Texture2D = texture.unwrap();
        let presentation = unsafe { self.manager.AddBufferFromResource(&texture) }?;
        let available = unsafe { Owned::new(presentation.GetAvailableEvent()?) };
        let shared = unsafe {
            Owned::new(texture.cast::<IDXGIResource1>()?.CreateSharedHandle(
                None,
                GENERIC_ALL.0,
                PCWSTR::null(),
            )?)
        };
        // Importing an NT handle doesn't take ownership of it, so `shared` still closes it.
        let image = PresentationImage::import(self.device.clone(), *shared, size, color_space);
        Ok(Buffer {
            image: GPUMutex::new(image),
            presentation,
            available,
        })
    }

    /// Presents `buffer` once `render_done` reaches `value`, and once the last
    /// present is queued for display. Returns false if the presentation manager is
    /// lost, and has to be replaced.
    ///
    /// Presents are FIFO: each is shown, in order. When several presents are ready
    /// at a compositor frame, the compositor shows the last and skips the others,
    /// so a present is ready only once the one before it is queued; it is then the
    /// only one ready. The wait is on the GPU, so the CPU goes on with the next
    /// frame.
    fn present(&mut self, buffer: &Buffer, value: u64) -> bool {
        unsafe {
            // A present is ready only once the work issued to the context before
            // the present is done: here, the waits for Vulkan's writes into the
            // buffer, and for the last present to be queued. The last present is
            // queued when the one before it starts retiring, which the retiring
            // fence counts. Present IDs start at 1, and the fence at 0.
            self.context.Wait(&self.render_done, value).unwrap();
            if let Some(before_last) = self.manager.GetNextPresentId().checked_sub(2) {
                self.context.Wait(&self.retiring, before_last).unwrap();
            }
            self.context.Flush();
            self.surface.SetBuffer(&buffer.presentation).unwrap();
            // Like the buffer, the color space changes with this present, so the
            // surface never shows a buffer in another color space.
            let color_space = buffer.image.color_space();
            if color_space != self.color_space {
                self.surface
                    .SetColorSpace(dxgi_color_space(color_space))
                    .unwrap();
                self.color_space = color_space;
            }
            // The source rect starts out empty, and nothing shows until it's set.
            // Buffers can differ in size, so it's set with each one.
            let extent = buffer.image.extent();
            self.surface
                .SetSourceRect(&RECT {
                    left: 0,
                    top: 0,
                    right: extent.x as i32,
                    bottom: extent.y as i32,
                })
                .unwrap();
            match self.manager.Present() {
                Ok(()) => true,
                // Only presents report a lost presentation manager.
                Err(error) if error.code() == PRESENTATION_ERROR_LOST => false,
                Err(error) => panic!("failed to present: {error}"),
            }
        }
    }
}

impl Drop for PresentationManager {
    fn drop(&mut self) {
        unsafe {
            // Submissions may still signal the semaphore, so this waits for the
            // device to be idle. A `PresentationManager` is dropped only at exit, or when its
            // presentation manager is lost.
            self.device.device_wait_idle().ok();
            self.device
                .destroy_semaphore(self.render_done_semaphore, None);
        }
    }
}

/// The DXGI color space that matches `color_space`, one of the two that
/// [`acquire`] picks from.
fn dxgi_color_space(color_space: vk::ColorSpaceKHR) -> DXGI_COLOR_SPACE_TYPE {
    match color_space {
        vk::ColorSpaceKHR::SRGB_NONLINEAR => DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709,
        vk::ColorSpaceKHR::HDR10_ST2084_EXT => DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020,
        _ => DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709,
    }
}

/// Creates the [`Swapchain`]'s presenter, once the Vulkan device exists. If the
/// system can't present this way, it removes the swapchain.
fn create_presenter(
    _bevy_thread: NonSendMarker,
    mut commands: Commands,
    device: Res<Device>,
    mut swapchain: ResMut<Swapchain>,
) {
    match PresentationManager::new(&device, swapchain.surface) {
        Ok(presenter) => swapchain.presenter = Some(presenter),
        Err(error) => {
            tracing::error!("The game can't present: {error}");
            commands.remove_resource::<Swapchain>();
        }
    }
}

/// Picks the next buffer once the compositor is done with it, makes it again if
/// the game's size or color space has changed, and makes it the current image.
/// The submission that this system and [`SwapchainSet`] are in signals
/// `render_done` once it has written the buffer.
fn acquire(
    _bevy_thread: NonSendMarker,
    mut submission: SubmissionState,
    If(swapchain): If<ResMut<Swapchain>>,
) {
    let swapchain = swapchain.into_inner();
    let Swapchain {
        presenter: Some(presenter),
        buffers,
        game_layer_state,
        index,
        current,
        state,
        ..
    } = swapchain
    else {
        return;
    };
    // Once the wait times out, the next frame tries the same buffer again.
    let next = (*index + 1) % BUFFER_COUNT;
    if let Some(buffer) = &buffers[next]
        && !presenter.wait_available(buffer, 100)
    {
        return;
    }
    *index = next;

    // Buffers can differ in size and color space, so the others stay on screen
    // until this one is presented, and a change reaches all of them in
    // `BUFFER_COUNT` frames.
    let size = game_layer_state.size();
    presenter.read_statistics();
    let color_space = presenter.preferred_color_space;
    let buffer = &mut buffers[next];
    if buffer.as_ref().is_none_or(|buffer| {
        buffer.image.extent().truncate() != size || buffer.image.color_space() != color_space
    }) {
        match presenter.create_buffer(size, color_space) {
            Ok(new) => *buffer = Some(new),
            Err(error) => panic!("failed to create a presentation buffer: {error}"),
        }
    }

    presenter.render_done_value += 1;
    let value = presenter.render_done_value;
    // SAFETY: the semaphore outlives the submission: it's destroyed only after the
    // device is idle. Only this submission signals it, with increasing values.
    unsafe {
        submission.signal_semaphore(
            presenter.render_done_semaphore,
            value,
            vk::PipelineStageFlags2::ALL_COMMANDS,
        );
    }
    *current = Some(value);
    // The buffer's earlier contents are discarded, so it needn't be acquired from
    // D3D11 (`VK_QUEUE_FAMILY_EXTERNAL`) first. Its first barrier waits for all the
    // commands before it in the submission, as on macOS.
    *state = ResourceState::default();
    state.write.stage = vk::PipelineStageFlags2::ALL_COMMANDS;
}

/// Hands the current image over to D3D11, after [`SwapchainSet`] is done with it.
fn release(
    mut submission: SubmissionState,
    queue: Queue<RenderQueue>,
    If(swapchain): If<ResMut<Swapchain>>,
) {
    let swapchain = swapchain.into_inner();
    submission.record(|encoder| {
        let Some(image) = swapchain.current_image() else {
            return;
        };
        let image = encoder.lock(image, vk::PipelineStageFlags2::ALL_COMMANDS);
        let state = swapchain.state_mut();
        encoder.image_ownership_transfer(
            image,
            Access {
                stage: vk::PipelineStageFlags2::ALL_COMMANDS,
                access: vk::AccessFlags2::MEMORY_WRITE,
            },
            Access::NONE,
            state.layout,
            vk::ImageLayout::GENERAL,
            queue.family_index(),
            vk::QUEUE_FAMILY_EXTERNAL,
            0..1,
            0..1,
        );
        state.layout = vk::ImageLayout::GENERAL;
        encoder.emit_barriers();
    });
}

/// Presents the current image once the submission of [`SwapchainSet`] is done
/// writing it.
fn present(
    _bevy_thread: NonSendMarker,
    mut commands: Commands,
    If(swapchain): If<ResMut<Swapchain>>,
) {
    let swapchain = swapchain.into_inner();
    let Some(value) = swapchain.current.take() else {
        return;
    };
    let (Some(presenter), Some(buffer)) = (
        &mut swapchain.presenter,
        &swapchain.buffers[swapchain.index],
    ) else {
        return;
    };
    if !presenter.present(buffer, value)
        && let Err(error) = swapchain.replace_lost_presenter()
    {
        tracing::error!("The game can't present: {error}");
        commands.remove_resource::<Swapchain>();
    }
}

/// A presentation buffer's texture imported as a `VkImage`, with a view of all of
/// it: the image a frame renders into.
pub struct PresentationImage {
    device: Device,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: PresentationImageView,
    extent: UVec2,
    color_space: vk::ColorSpaceKHR,
}

/// The texture's format: `DXGI_FORMAT_R10G10B10A2_UNORM`, which Vulkan calls
/// `A2B10G10R10_UNORM_PACK32`. The macOS layer uses the same format.
const FORMAT: vk::Format = vk::Format::A2B10G10R10_UNORM_PACK32;

impl PresentationImage {
    /// Imports the texture behind `shared`, an NT handle from
    /// `IDXGIResource1::CreateSharedHandle`, as an image that the visual shows in
    /// `color_space`.
    fn import(
        device: Device,
        shared: HANDLE,
        extent: UVec2,
        color_space: vk::ColorSpaceKHR,
    ) -> Self {
        let handle_type = vk::ExternalMemoryHandleTypeFlags::D3D11_TEXTURE;
        let mut external = vk::ExternalMemoryImageCreateInfo::default().handle_types(handle_type);
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(FORMAT)
            .extent(vk::Extent3D {
                width: extent.x,
                height: extent.y,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push(&mut external);
        let image = unsafe { device.create_image(&info, None) }.unwrap();
        let mut handle_properties = vk::MemoryWin32HandlePropertiesKHR::default();
        unsafe {
            device
                .extension::<khr::external_memory_win32::Meta>()
                .get_memory_win32_handle_properties(
                    handle_type,
                    shared.0 as vk::HANDLE,
                    &mut handle_properties,
                )
        }
        .unwrap();
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let memory_types = handle_properties.memory_type_bits & requirements.memory_type_bits;
        assert_ne!(
            memory_types, 0,
            "no memory type can hold the imported texture"
        );
        // The memory backing the image is the texture itself. Vulkan can only
        // import a D3D11 texture as a dedicated allocation.
        let mut import = vk::ImportMemoryWin32HandleInfoKHR::default()
            .handle_type(handle_type)
            .handle(shared.0 as vk::HANDLE);
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);
        let memory = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(memory_types.trailing_zeros())
                    .push(&mut import)
                    .push(&mut dedicated),
                None,
            )
        }
        .unwrap();
        unsafe { device.bind_image_memory(image, memory, 0) }.unwrap();
        let view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(FORMAT)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
                None,
            )
        }
        .unwrap();
        PresentationImage {
            device,
            image,
            memory,
            view: PresentationImageView(view),
            extent,
            color_space,
        }
    }

    /// A view of the whole image, in its own format.
    pub fn linear_view(&self) -> &PresentationImageView {
        &self.view
    }

    /// The color space the visual shows the image in: HDR10 while the display
    /// is in HDR mode, and sRGB otherwise.
    pub fn color_space(&self) -> vk::ColorSpaceKHR {
        self.color_space
    }
}

project_host_metadata! {
    /// [`PresentationImage`]'s accessors on a `GPUMutex` holding it, and on the
    /// tokens locking it.
    pub trait PresentationImageMetadata for impl[] PresentationImage {
        /// A view of the whole image, in its own format.
        fn linear_view(&self) -> &PresentationImageView;
        /// The color space the visual shows the image in.
        fn color_space(&self) -> vk::ColorSpaceKHR;
    }
}

impl AsVkHandle for PresentationImage {
    type Handle = vk::Image;

    fn vk_handle(&self) -> Self::Handle {
        self.image
    }
}

// SAFETY: one `PresentationImage` names each imported image, which it creates and
// destroys. It isn't `Clone`.
unsafe impl ImageLike for PresentationImage {
    fn aspects(&self) -> vk::ImageAspectFlags {
        vk::ImageAspectFlags::COLOR
    }

    fn array_layer_count(&self) -> u32 {
        1
    }

    fn mip_level_count(&self) -> u32 {
        1
    }

    fn extent(&self) -> UVec3 {
        self.extent.extend(1)
    }

    fn format(&self) -> vk::Format {
        FORMAT
    }

    fn ty(&self) -> vk::ImageType {
        vk::ImageType::TYPE_2D
    }
}

impl Drop for PresentationImage {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_image_view(self.view.0, None);
            self.device.destroy_image(self.image, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

/// The view [`PresentationImage::linear_view`] returns.
pub struct PresentationImageView(vk::ImageView);

impl AsVkHandle for PresentationImageView {
    type Handle = vk::ImageView;

    fn vk_handle(&self) -> Self::Handle {
        self.0
    }
}
