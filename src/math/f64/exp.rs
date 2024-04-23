use std::{
    f64::consts::{LN_2, LOG2_E},
    simd::{prelude::*, LaneCount, StdFloat, SupportedLaneCount},
};

use crate::{math::Exponent, polynomial_simd};

fn exp_m1_taylor<const N: usize>(x: Simd<f64, N>) -> Simd<f64, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    const P2: f64 = 1.0 / 2.0;
    const P3: f64 = 1.0 / 6.0;
    const P4: f64 = 1.0 / 24.0;
    const P5: f64 = 1.0 / 120.0;
    const P6: f64 = 1.0 / 720.0;
    const P7: f64 = 1.0 / 5040.0;
    const P8: f64 = 1.0 / 40320.0;
    const P9: f64 = 1.0 / 362880.0;
    const P10: f64 = 1.0 / 3628800.0;
    const P11: f64 = 1.0 / 39916800.0;
    const P12: f64 = 1.0 / 479001600.0;
    const P13: f64 = 1.0 / 6227020800.0;

    polynomial_simd!(x; P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13).mul_add(x * x, x)
}

/// Calculates `pow(2.0, n)` where `n` must be an integer.
/// Does not check for overflow or underflow.
#[inline]
fn pow2i<const N: usize>(n: Simd<f64, N>) -> Simd<f64, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    const MANTISSA_BITS: f64 = (1u64 << (f64::MANTISSA_DIGITS - 1)) as f64;
    const BIAS: f64 = 1023.0;
    Simd::from_bits(
        Simd::to_bits(n + Simd::splat(BIAS + MANTISSA_BITS)) << (f64::MANTISSA_DIGITS as u64 - 1),
    )
}

/// Returns reduced argument `r` and an integer power of 2.0 `n`, s.t. `exp(x) = exp(r) * n`.
#[inline]
fn exp_arg_reduction<const N: usize>(x: Simd<f64, N>) -> (Simd<f64, N>, Simd<f64, N>)
where
    LaneCount<N>: SupportedLaneCount,
{
    // src: https://github.com/vectorclass/version2/blob/master/vectormath_exp.h#L168-L169
    const LN2_HI: f64 = 0.693_145_751_953_125;
    const LN2_LO: f64 = 1.428_606_820_309_417_3E-6;

    let n = (x * Simd::splat(LOG2_E)).round();
    let reduced_x = n.mul_add(Simd::splat(-LN2_LO), n.mul_add(Simd::splat(-LN2_HI), x));
    (reduced_x, pow2i(n))
}

// using macro instead of function here because `f64` cannot be passed as a generic parameter
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
                    Simd::splat(f64::NAN),
                    x.is_sign_positive()
                        .select(Simd::splat(f64::INFINITY), Simd::splat($NEGINF_VAL)),
                ),
            )
        }
    }};
}

impl<const N: usize> Exponent for Simd<f64, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline]
    fn exp(self) -> Self {
        let (reduced, n) = exp_arg_reduction(self);
        exp_handle_overflow_and_special!(
            709.0,
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
            1023.0,
            0.0,
            self,
            (exp_m1_taylor(reduced) + Simd::splat(1.0)) * pow2i(r)
        )
    }

    #[inline]
    fn exp_m1(self) -> Self {
        let (reduced, n) = exp_arg_reduction(self);
        exp_handle_overflow_and_special!(
            709.0,
            -1.0,
            self,
            exp_m1_taylor(reduced).mul_add(n, n - Simd::splat(1.0))
        )
    }
}
