use std::simd::{num::SimdFloat, LaneCount, Simd, SupportedLaneCount};

/// Returns `1u32 << 31` if sign-bit is set and zero otherwise.
///
/// ```
/// # #![feature(portable_simd)]
/// # use std::simd::prelude::*;
/// # use simd_addons::math::util::*;
/// assert_eq!(
///     sign_mask_f32(f32x4::from_array([0.0, -0.0, 1.0, -f32::INFINITY])),
///     u32x4::from_array([0, 1, 0, 1]) << 31
/// )
/// ```
#[inline]
pub fn sign_mask_f32<const N: usize>(value: Simd<f32, N>) -> Simd<u32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    value.to_bits() & Simd::splat(1 << 31)
}

/// Returns `1u64 << 63` if sign-bit is set and zero otherwise.
///
/// ```
/// # #![feature(portable_simd)]
/// # use std::simd::prelude::*;
/// # use simd_addons::math::util::*;
/// assert_eq!(
///     sign_mask_f64(f64x4::from_array([0.0, -0.0, 1.0, f64::NEG_INFINITY])),
///     u64x4::from_array([0, 1, 0, 1]) << 63
/// )
/// ```
#[inline]
pub fn sign_mask_f64<const N: usize>(value: Simd<f64, N>) -> Simd<u64, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    value.to_bits() & Simd::splat(1 << 63)
}
