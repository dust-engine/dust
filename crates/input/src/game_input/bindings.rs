//! Bindings for the parts of GameInput's API (version 4, `GameInput.h` in the
//! `Microsoft.GameInput` NuGet package) that the backend uses, with the
//! header's names.
//!
//! The interfaces declare the methods in vtable order up to the last one used;
//! the methods after it are left out.

#![allow(non_snake_case, non_upper_case_globals, dead_code)]

use std::ffi::c_void;

use windows::Win32::System::LibraryLoader::{
    GetProcAddress, LOAD_LIBRARY_SEARCH_SYSTEM32, LoadLibraryExW,
};
use windows::core::{s, w};
use windows_core::{Error, GUID, HRESULT, IUnknown, IUnknown_Vtbl, Interface, Result, interface};

pub type GameInputKind = u32;
pub const GameInputKindKeyboard: GameInputKind = 0x10;
pub const GameInputKindMouse: GameInputKind = 0x20;

pub type GameInputMouseButtons = u32;
pub const GameInputMouseNone: GameInputMouseButtons = 0x00;
pub const GameInputMouseLeftButton: GameInputMouseButtons = 0x01;
pub const GameInputMouseRightButton: GameInputMouseButtons = 0x02;
pub const GameInputMouseMiddleButton: GameInputMouseButtons = 0x04;
pub const GameInputMouseButton4: GameInputMouseButtons = 0x08;
pub const GameInputMouseButton5: GameInputMouseButtons = 0x10;

pub type GameInputMousePositions = u32;
pub const GameInputMouseRelativePosition: GameInputMousePositions = 0x02;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GameInputKeyState {
    pub scanCode: u32,
    pub codePoint: u32,
    pub virtualKey: u8,
    pub isDeadKey: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GameInputMouseState {
    pub buttons: GameInputMouseButtons,
    pub positions: GameInputMousePositions,
    pub positionX: i64,
    pub positionY: i64,
    pub absolutePositionX: i64,
    pub absolutePositionY: i64,
    pub wheelX: i64,
    pub wheelY: i64,
}

#[interface("F2DAAE4C-4D21-4250-8953-4F791E6279AB")]
pub unsafe trait IGameInput: IUnknown {
    pub fn GetCurrentTimestamp(&self) -> u64;
    pub fn GetCurrentReading(
        &self,
        inputKind: GameInputKind,
        device: *mut c_void,
        reading: *mut Option<IGameInputReading>,
    ) -> HRESULT;
    pub fn GetNextReading(
        &self,
        referenceReading: *mut c_void,
        inputKind: GameInputKind,
        device: *mut c_void,
        reading: *mut Option<IGameInputReading>,
    ) -> HRESULT;
    pub fn GetPreviousReading(
        &self,
        referenceReading: *mut c_void,
        inputKind: GameInputKind,
        device: *mut c_void,
        reading: *mut Option<IGameInputReading>,
    ) -> HRESULT;
}

#[interface("0DFAA2A2-688B-4A9C-A3A9-0FEB4F90EE12")]
pub unsafe trait IGameInputReading: IUnknown {
    pub fn GetInputKind(&self) -> GameInputKind;
    pub fn GetTimestamp(&self) -> u64;
    /// Returns a reference the caller releases.
    pub fn GetDevice(&self, device: *mut Option<IGameInputDevice>);
    pub fn GetControllerAxisCount(&self) -> u32;
    pub fn GetControllerAxisState(&self, stateArrayCount: u32, stateArray: *mut f32) -> u32;
    pub fn GetControllerButtonCount(&self) -> u32;
    pub fn GetControllerButtonState(&self, stateArrayCount: u32, stateArray: *mut bool) -> u32;
    pub fn GetControllerSwitchCount(&self) -> u32;
    pub fn GetControllerSwitchState(&self, stateArrayCount: u32, stateArray: *mut u32) -> u32;
    pub fn GetKeyCount(&self) -> u32;
    pub fn GetKeyState(&self, stateArrayCount: u32, stateArray: *mut GameInputKeyState) -> u32;
    pub fn GetMouseState(&self, state: *mut GameInputMouseState) -> bool;
}

#[interface("4420F01D-8918-49D4-B302-DBEAD986105F")]
pub unsafe trait IGameInputDevice: IUnknown {}

// SAFETY: GameInput's objects are thread-safe: they may be used from any
// thread, at the same time.
unsafe impl Send for IGameInput {}
unsafe impl Sync for IGameInput {}
unsafe impl Send for IGameInputDevice {}
unsafe impl Sync for IGameInputDevice {}

type GameInputInitialize =
    unsafe extern "system" fn(riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT;

/// Creates GameInput, from the redistributable in System32 (installed with
/// `winget install Microsoft.GameInput`), as `GameInputCreate` does. The
/// redistributable stays loaded.
pub fn create_game_input() -> Result<IGameInput> {
    unsafe {
        let redistributable = LoadLibraryExW(
            w!("GameInputRedist.dll"),
            None,
            LOAD_LIBRARY_SEARCH_SYSTEM32,
        )?;
        let game_input_initialize = GetProcAddress(redistributable, s!("GameInputInitialize"))
            .ok_or_else(Error::from_thread)?;
        let game_input_initialize: GameInputInitialize = std::mem::transmute(game_input_initialize);
        let mut game_input = std::ptr::null_mut();
        game_input_initialize(&IGameInput::IID, &mut game_input).ok()?;
        Ok(IGameInput::from_raw(game_input))
    }
}
