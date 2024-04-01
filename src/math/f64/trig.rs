use std::{
    f64::consts::{FRAC_2_PI, FRAC_PI_2},
    simd::{prelude::*, LaneCount, StdFloat, SupportedLaneCount},
};

use crate::{
    math::{util::FloatBitUtils, Trigonometry},
    polynomial_simd,
};

#[inline]
fn trig_reduction<const N: usize>(x: Simd<f64, N>) -> (Simd<f64, N>, Simd<u64, N>)
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

    // SAFETY: values in `quadrants_float` are finite and between 0 and 1e13
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
fn sin_cos_taylor<const N: usize>(x: Simd<f64, N>) -> (Simd<f64, N>, Simd<f64, N>)
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
    #[inline]
    fn sin(self) -> Self {
        let (reduced_x, quadrants) = trig_reduction(self);
        let (sin, cos) = sin_cos_taylor(reduced_x);

        let sin_cos_swap = (quadrants & Simd::splat(1)).simd_eq(Simd::default());
        let sin_vals = sin_cos_swap.select(sin, cos);
        sin_vals
            .sign_combine(self)
            .sign_combine(Simd::from_bits(quadrants << 62))
    }

    #[inline]
    fn cos(self) -> Self {
        let (reduced_x, quadrants) = trig_reduction(self);
        let (sin, cos) = sin_cos_taylor(reduced_x);

        let sin_cos_swap = (quadrants & Simd::splat(1)).simd_eq(Simd::default());
        let cos_vals = sin_cos_swap.select(cos, sin);
        cos_vals.sign_combine(Simd::from_bits((quadrants + Simd::splat(1)) << 62))
    }

    #[inline]
    fn tan(self) -> Self {
        const P0: f64 = -1.795_652_519_764_848_8E7;
        const P1: f64 = 1.153_516_648_385_874_2E6;
        const P2: f64 = -1.309_369_391_813_837_9E4;

        const Q0: f64 = -5.386_957_559_294_546_4E7;
        const Q1: f64 = 2.500_838_018_233_579E7;
        const Q2: f64 = -1.320_892_344_402_109_7E6;
        const Q3: f64 = 1.368_129_634_706_929_6E4;

        let (reduced_x, quadrants) = trig_reduction(self);
        let reduced_x2 = reduced_x * reduced_x;
        let tan_vals = (polynomial_simd!(reduced_x2; P0, P1, P2)
            / polynomial_simd!(reduced_x2; Q0, Q1, Q2, Q3, 1.0))
        .mul_add(reduced_x * reduced_x2, reduced_x);

        (quadrants & Simd::splat(1))
            .simd_eq(Simd::default())
            .select(tan_vals, -tan_vals.recip())
            .sign_combine(self)
    }

    fn asin(self) -> Self {
        const P0: f64 = -8.198_089_802_484_825;
        const P1: f64 = 1.956_261_983_317_594_8E1;
        const P2: f64 = -1.626_247_967_210_700_2E1;
        const P3: f64 = 5.444_622_390_564_711;
        const P4: f64 = -6.019_598_008_014_124E-1;
        const P5: f64 = 4.253_011_369_004_428E-3;

        const Q0: f64 = -4.918_853_881_490_881E1;
        const Q1: f64 = 1.395_105_614_657_485_7E2;
        const Q2: f64 = -1.471_791_292_232_726E2;
        const Q3: f64 = 7.049_610_280_856_842E1;
        const Q4: f64 = -1.474_091_372_988_853_8E1;

        let x_abs = self.abs();
        let big = x_abs.simd_ge(Simd::splat(0.5));

        // for x >= 0: π/2 - 2 arcsin √((1-x)/2) = arcsin x
        // taylor_arg is less than or equal to 0.5
        let pade_arg2 = big.select((Simd::splat(1.0) - x_abs) * Simd::splat(0.5), x_abs * x_abs);
        let pade_arg = big.select(pade_arg2.sqrt(), x_abs);
        let pade_result = (polynomial_simd!(pade_arg2; P0, P1, P2, P3, P4, P5)
            / polynomial_simd!(pade_arg2; Q0, Q1, Q2, Q3, Q4, 1.0))
        .mul_add(pade_arg2 * pade_arg, pade_arg);

        let asin_abs = big.select(
            Simd::splat(FRAC_PI_2) - (pade_result + pade_result),
            pade_result,
        );
        asin_abs.sign_combine(self)
    }

    fn acos(self) -> Self {
        Simd::splat(FRAC_PI_2) - self.asin()
    }

    fn atan(self) -> Self {
        todo!()
    }

    fn atan2(self, _x: Self) -> Self {
        todo!()
    }
}
