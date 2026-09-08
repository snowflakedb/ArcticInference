//! The one place in the crate where a lossy numeric conversion may use `as`.
//!
//! `as_conversions` is denied crate-wide. Most casts have a faithful replacement
//! (`u64::from`, `u32::try_from`) belonging at the call site. These do not: the
//! stdlib provides no `From`/`TryFrom` for int→float above the mantissa width, nor
//! for float→int. Centralizing them here keeps the escape hatch reviewed once, in
//! named functions that state the loss.
//!
//! Float→int relies on Rust's saturating cast semantics (since 1.45): clamped at
//! the bounds, `NaN` → 0. Total, no UB, no panic.

/// `u64` → `f64`, losing precision above 2^53. For statistics and display, where
/// the input is a count or a duration that never approaches that magnitude.
#[must_use]
#[expect(clippy::as_conversions, clippy::cast_precision_loss, reason = "no From<u64> for f64")]
pub const fn u64_to_f64_lossy(x: u64) -> f64 {
    x as f64
}

/// `usize` → `f64`, losing precision above 2^53. Chiefly `len()` feeding a mean,
/// ratio, or percentage.
#[must_use]
#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "no From<usize> for f64"
)]
pub const fn usize_to_f64_lossy(x: usize) -> f64 {
    x as f64
}

/// `u32` → `f32`, losing precision above 2^24. Image dimension arithmetic.
#[must_use]
#[expect(clippy::as_conversions, clippy::cast_precision_loss, reason = "no From<u32> for f32")]
pub const fn u32_to_f32_lossy(x: u32) -> f32 {
    x as f32
}

/// `f64` → `u64`, saturating at the bounds; `NaN` becomes 0. Truncates toward
/// zero — round first if you want nearest.
#[must_use]
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "saturating cast is total (Rust 1.45)"
)]
pub const fn f64_to_u64_saturating(x: f64) -> u64 {
    x as u64
}

/// `f64` → `u32`, saturating at the bounds; `NaN` becomes 0. Truncates toward
/// zero — round first if you want nearest.
#[must_use]
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "saturating cast is total (Rust 1.45)"
)]
pub const fn f64_to_u32_saturating(x: f64) -> u32 {
    x as u32
}

/// `f32` → `u32`, saturating at the bounds; `NaN` becomes 0. Truncates toward
/// zero — round first if you want nearest.
#[must_use]
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "saturating cast is total (Rust 1.45)"
)]
pub const fn f32_to_u32_saturating(x: f32) -> u32 {
    x as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_to_int_saturates_instead_of_wrapping() {
        assert_eq!(f64_to_u32_saturating(-1.0), 0);
        assert_eq!(f64_to_u32_saturating(f64::NAN), 0);
        assert_eq!(f64_to_u32_saturating(1e300), u32::MAX);
        assert_eq!(f64_to_u64_saturating(-0.5), 0);
        assert_eq!(f64_to_u64_saturating(f64::INFINITY), u64::MAX);
        assert_eq!(f32_to_u32_saturating(f32::NAN), 0);
    }

    #[test]
    fn float_to_int_truncates_toward_zero() {
        assert_eq!(f64_to_u32_saturating(3.9), 3);
        assert_eq!(f64_to_u64_saturating(0.999), 0);
    }

    #[test]
    fn int_to_float_is_exact_below_the_mantissa_width() {
        assert!((usize_to_f64_lossy(1 << 52) - 4_503_599_627_370_496.0).abs() < f64::EPSILON);
        assert!((u64_to_f64_lossy(0) - 0.0).abs() < f64::EPSILON);
        assert!((u32_to_f32_lossy(1 << 23) - 8_388_608.0).abs() < f32::EPSILON);
    }
}
