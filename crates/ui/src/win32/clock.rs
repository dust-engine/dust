//! The system's clocks, as the frame timeline takes times: nanoseconds from now
//! (see `dust_profiler::timeline`).

use std::sync::OnceLock;

use windows::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};
use windows::Win32::System::WindowsProgramming::QueryInterruptTimePrecise;

/// How far `qpc`, a `QueryPerformanceCounter` time, is from now.
pub(crate) fn qpc_offset_ns(qpc: u64) -> i64 {
    let mut now = 0;
    unsafe { QueryPerformanceCounter(&mut now) }.ok();
    qpc_ns(qpc as i128 - i128::from(now))
}

/// `ticks` of the `QueryPerformanceCounter`, in nanoseconds.
pub(crate) fn qpc_ns(ticks: impl Into<i128>) -> i64 {
    static FREQUENCY: OnceLock<i64> = OnceLock::new();
    let frequency = *FREQUENCY.get_or_init(|| {
        let mut frequency = 0;
        unsafe { QueryPerformanceFrequency(&mut frequency) }.ok();
        frequency.max(1)
    });
    (ticks.into() * 1_000_000_000 / i128::from(frequency)) as i64
}

/// How far `interrupt_time`, an interrupt time in 100 ns units such as the
/// presentation API's `SystemInterruptTime`, is from now.
pub(crate) fn interrupt_time_offset_ns(interrupt_time: u64) -> i64 {
    let now = unsafe { QueryInterruptTimePrecise() };
    (interrupt_time as i64 - now as i64) * 100
}
