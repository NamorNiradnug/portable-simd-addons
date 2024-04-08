use crate::{math::Exponent, polynomial_simd};
use std::{
    f32::consts::*,
    simd::{num::SimdFloat, LaneCount, Simd, StdFloat, SupportedLaneCount},
};

#[inline]
fn exp_m1_taylor<const N: usize>(x: Simd<f32, N>) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    const P0: f32 = 1.0 / 2.0;
    const P1: f32 = 1.0 / 6.0;
    const P2: f32 = 1.0 / 24.0;
    const P3: f32 = 1.0 / 120.0;
    const P4: f32 = 1.0 / 720.0;
    const P5: f32 = 1.0 / 5040.0;

    polynomial_simd!(x; P0, P1, P2, P3, P4, P5).mul_add(x * x, x)
}

/// Calculates `pow(2.0, n)` where `n` must be an integer.
/// Does not check for overflow or underflow.
#[inline]
fn pow2i<const N: usize>(n: Simd<f32, N>) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    const POW2_23: f32 = 8388608.0;
    const BIAS: f32 = 127.0;
    Simd::from_bits(Simd::to_bits(n + Simd::splat(BIAS + POW2_23)) << (f32::MANTISSA_DIGITS - 1))
}

impl<const N: usize> Exponent for Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline]
    fn exp(self) -> Self {
        todo!()
    }

    #[inline]
    fn exp2(self) -> Self {
        todo!()
    }

    #[inline]
    fn exp_m1(self) -> Self {
        const LN2_HI: f32 = 0.693_359_4;
        const LN2_LO: f32 = 2.121_944_4e-4;

        // exp(self) = exp(reduced) * 2^r
        let r = (self * Simd::splat(LOG2_E)).round();
        let reduced_x = r.mul_add(Simd::splat(-LN2_LO), r.mul_add(Simd::splat(-LN2_HI), self));

        let pow2_r = pow2i(r);
        exp_m1_taylor(reduced_x).mul_add(pow2_r, pow2_r - Simd::splat(1.0))
    }
}
