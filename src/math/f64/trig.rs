use std::{
    f64::consts::FRAC_2_PI,
    simd::{prelude::*, LaneCount, StdFloat, SupportedLaneCount},
};

use crate::{
    math::{util::FloatBitUtils, Trigonometry},
    polynomial_simd,
};

#[inline]
fn trig_reduction_f64<const N: usize>(x: Simd<f64, N>) -> (Simd<f64, N>, Simd<u64, N>)
where
    LaneCount<N>: SupportedLaneCount,
{
    const INPUT_LIMIT: f64 = 1e13;

    const PI2_A: f64 = 7.853_981_554_508_209E-1 * 2.;
    const PI2_B: f64 = 7.946_627_356_147_928E-9 * 2.;
    const PI2_C: f64 = 3.061_616_997_868_383E-17 * 2.;

    let mut abs_x = x.abs();
    // NaNs, INFs and large values are mapped to 0
    abs_x = abs_x
        .simd_lt(Simd::splat(INPUT_LIMIT))
        .select(abs_x, Simd::default());
    let quadrants_float = (abs_x * Simd::splat(FRAC_2_PI)).round();

    // SAFETY: values in `quadrants_float` are finite and between 0 and 1e5
    let quadrants = unsafe { quadrants_float.to_int_unchecked::<u64>() };

    let reduced_x = quadrants_float.mul_add(
        Simd::splat(-PI2_C),
        quadrants_float.mul_add(
            Simd::splat(-PI2_B),
            quadrants_float.mul_add(Simd::splat(-PI2_A), abs_x),
        ),
    );

    (reduced_x, quadrants)
}

#[inline]
fn sin_cos_taylor_f64<const N: usize>(x: Simd<f64, N>) -> (Simd<f64, N>, Simd<f64, N>)
where
    LaneCount<N>: SupportedLaneCount,
{
    const P0_SIN: f64 = -1.666_666_666_666_663E-1;
    const P1_SIN: f64 = 8.333_333_333_322_118E-3;
    const P2_SIN: f64 = -1.984_126_982_958_954E-4;
    const P3_SIN: f64 = 2.755_731_362_138_572_2E-6;
    const P4_SIN: f64 = -2.505_074_776_285_780_7E-8;
    const P5_SIN: f64 = 1.589_623_015_765_465_6E-10;

    const P0_COS: f64 = 4.166_666_666_666_659_5E-2;
    const P1_COS: f64 = -1.388_888_888_887_305_6E-3;
    const P2_COS: f64 = 2.480_158_728_885_170_4E-5;
    const P3_COS: f64 = -2.755_731_417_929_674E-7;
    const P4_COS: f64 = 2.087_570_084_197_473E-9;
    const P5_COS: f64 = -1.135_853_652_138_768_2E-11;

    let x2 = x * x;
    let sin =
        polynomial_simd!(x2; P0_SIN, P1_SIN, P2_SIN, P3_SIN, P4_SIN, P5_SIN).mul_add(x2 * x, x);
    let cos = polynomial_simd!(x2; 1.0, -0.5, P0_COS, P1_COS, P2_COS, P3_COS, P4_COS, P5_COS);

    (sin, cos)
}

impl<const N: usize> Trigonometry for Simd<f64, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    fn sin(self) -> Self {
        let (reduced_x, quadrants) = trig_reduction_f64(self);
        let (sin, cos) = sin_cos_taylor_f64(reduced_x);

        let sin_cos_swap = (quadrants & Simd::splat(1)).simd_eq(Simd::default());
        let sin_vals = sin_cos_swap.select(sin, cos);
        sin_vals
            .sign_combine(self)
            .sign_combine(Simd::from_bits((quadrants & Simd::splat(1)) << 62))
    }

    fn cos(self) -> Self {
        let (reduced_x, quadrants) = trig_reduction_f64(self);
        let (sin, cos) = sin_cos_taylor_f64(reduced_x);

        let sin_cos_swap = (quadrants & Simd::splat(1)).simd_eq(Simd::default());
        let cos_vals = sin_cos_swap.select(cos, sin);
        cos_vals.sign_combine(Simd::from_bits((quadrants + Simd::splat(1)) << 62))
    }

    fn asin(self) -> Self {
        todo!()
    }

    fn acos(self) -> Self {
        todo!()
    }

    fn atan(self) -> Self {
        todo!()
    }

    fn atan2(self, _x: Self) -> Self {
        todo!()
    }
}
