#[inline]
pub fn div_round_up(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}
