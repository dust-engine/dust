//! Crate root for `//:dust_static`, the staticlib linked into the macOS app
//! bundle (`//macos:Dust`). `macos_application` links the executable itself
//! from `CcInfo` deps, so the entry point is exported as a C `main` for the
//! linker to pull out of the archive.

#[unsafe(no_mangle)]
pub extern "C" fn main(
    _argc: core::ffi::c_int,
    _argv: *const *const core::ffi::c_char,
) -> core::ffi::c_int {
    dust::run();
    0
}
