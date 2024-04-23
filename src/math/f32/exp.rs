use crate::{math::Exponent, polynomial_simd};
use std::{
    f32::consts::*,
    simd::{prelude::*, LaneCount, StdFloat, SupportedLaneCount},
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
    const POW2_23: f32 = (1 << (f32::MANTISSA_DIGITS - 1)) as f32;
    const BIAS: f32 = 127.0;
    Simd::from_bits(Simd::to_bits(n + Simd::splat(BIAS + POW2_23)) << (f32::MANTISSA_DIGITS - 1))
}

/// Returns reduced argument `r` and an integer power of 2.0 `n`, s.t. `exp(x) = exp(r) * n`.
#[inline]
fn exp_arg_reduction<const N: usize>(x: Simd<f32, N>) -> (Simd<f32, N>, Simd<f32, N>)
where
    LaneCount<N>: SupportedLaneCount,
{
    const LN2_HI: f32 = 0.693_359_4;
    const LN2_LO: f32 = 2.121_944_4e-4;

    let n = (x * Simd::splat(LOG2_E)).round();
    let reduced_x = n.mul_add(Simd::splat(LN2_LO), n.mul_add(Simd::splat(-LN2_HI), x));
    (reduced_x, pow2i(n))
}

// using macro instead of function here because `f32` cannot be passed as a generic parameter
macro_rules! exp_handle_overflow_and_special {
    ($LIMIT: literal, $NEGINF_VAL: literal, $x: expr, $exp: expr) => {{
        let x = $x;
        let exp = $exp;
        let in_range = x.abs().simd_le(Simd::splat($LIMIT));
        if in_range.all() {
            exp
        } else {
            in_range.select(
                exp,
                x.is_nan().select(
                    Simd::splat(f32::NAN),
                    x.is_sign_positive()
                        .select(Simd::splat(f32::INFINITY), Simd::splat($NEGINF_VAL)),
                ),
            )
        }
    }};
}

impl<const N: usize> Exponent for Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline]
    fn exp(self) -> Self {
        let (reduced, n) = exp_arg_reduction(self);
        exp_handle_overflow_and_special!(
            88.0,
            0.0,
            self,
            (exp_m1_taylor(reduced) + Simd::splat(1.0)) * n
        )
    }

    #[inline]
    fn exp2(self) -> Self {
        let r = self.round();
        let reduced = (self - r) * Simd::splat(LN_2);
        exp_handle_overflow_and_special!(
            127.0,
            0.0,
            self,
            (exp_m1_taylor(reduced) + Simd::splat(1.0)) * pow2i(r)
        )
    }

    #[inline]
    fn exp_m1(self) -> Self {
        let (reduced, n) = exp_arg_reduction(self);
        exp_handle_overflow_and_special!(
            88.0,
            -1.0,
            self,
            exp_m1_taylor(reduced).mul_add(n, n - Simd::splat(1.0))
        )
    }
}
