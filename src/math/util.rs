use std::{
    ops::BitXor,
    simd::{num::SimdFloat, LaneCount, Simd, SupportedLaneCount},
};

pub trait FloatBitUtils: SimdFloat {
    /// Returns zero if sign-bit of the value is zero or sign bit mask otherwise.
    ///
    /// # Examples
    /// ```
    /// # #![feature(portable_simd)]
    /// # use std::simd::prelude::*;
    /// # use simd_addons::math::util::*;
    /// assert_eq!(
    ///     f32x4::from_array([0.0, -0.0, 1.0, -f32::INFINITY]).sign_bit(),
    ///     u32x4::from_array([0, 1, 0, 1]) << 31
    /// );
    /// assert_eq!(
    ///     f64x4::from_array([0.0, -0.0, 1.0, -f64::INFINITY]).sign_bit(),
    ///     u64x4::from_array([0, 1, 0, 1]) << 63
    /// );
    /// ```
    fn sign_bit(self) -> Self::Bits;

    /// Returns each element with the magnitude `self` and sign combined from signs of `self` and `other`.
    /// Does roughly the same as `self * other.signum()`.
    ///
    /// # Examples
    /// ```
    /// # #![feature(portable_simd)]
    /// # use std::simd::prelude::*;
    /// # use simd_addons::math::util::*;
    /// let x = Simd::from_array([1.0, 0.0, -0.0, -2.0]);
    /// let signs = Simd::from_array([1.0, -2.0, -f32::INFINITY, -1.0]);
    /// assert_eq!(
    ///     x.sign_combine(signs),
    ///     Simd::from_array([1.0, -0.0, 0.0, 2.0])
    /// )
    /// ```
    #[inline]
    fn sign_combine(self, other: Self) -> Self
    where
        Self::Bits: BitXor<Output = Self::Bits>,
    {
        Self::from_bits(self.to_bits() ^ other.sign_bit())
    }
}

impl<const N: usize> FloatBitUtils for Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline]
    fn sign_bit(self) -> Simd<u32, N> {
        self.to_bits() & Simd::splat(1 << 31)
    }
}

impl<const N: usize> FloatBitUtils for Simd<f64, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline]
    fn sign_bit(self) -> Simd<u64, N> {
        self.to_bits() & Simd::splat(1 << 63)
    }
}

pub trait FastRound {
    /// Currently same as `round_ties_even`. Unlike `.round()`,
    /// compiles into a single rounding instruction on common platforms.
    fn fast_round(self) -> Self;
}

impl<const N: usize> FastRound for Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline]
    fn fast_round(self) -> Self {
        self.to_array().map(f32::round_ties_even).into()
    }
}

impl<const N: usize> FastRound for Simd<f64, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline]
    fn fast_round(self) -> Self {
        self.to_array().map(f64::round_ties_even).into()
    }
}
