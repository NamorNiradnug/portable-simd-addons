use crate::{
    math::{util::FloatBitUtils, Trigonometry},
    polynomial_simd,
};
use std::{
    f32::consts::*,
    simd::{
        cmp::{SimdPartialEq, SimdPartialOrd},
        num::{SimdFloat, SimdInt},
        LaneCount, Mask, Simd, StdFloat, SupportedLaneCount,
    },
};

/// π/4 reduction of `x`. Large values (greater than 10⁵) are treated as zeros.
#[inline]
fn trig_reduction<const N: usize>(x: Simd<f32, N>) -> (Simd<f32, N>, Simd<u32, N>)
where
    LaneCount<N>: SupportedLaneCount,
{
    // TODO: INPUT_LIMIT should be larger
    const INPUT_LIMIT: f32 = 1e5;

    // src: https://github.com/vectorclass/version2/blob/master/vectormath_trig.h#L241-L243
    const PI2_A: f32 = 0.785_156_25 * 2.0;
    const PI2_B: f32 = 2.418_756_5E-4 * 2.0;
    const PI2_C: f32 = 3.774_895E-8 * 2.0;

    let mut abs_x = x.abs();
    // NaNs, INFs and large values are mapped to 0
    abs_x = abs_x
        .simd_lt(Simd::splat(INPUT_LIMIT))
        .select(abs_x, Simd::default());
    let quadrants_float = (abs_x * Simd::splat(FRAC_2_PI)).round();

    // SAFETY: INPUT_LIMIT guaratees that `quadrants_float` are representable in u32
    let quadrants = unsafe { quadrants_float.to_int_unchecked::<i32>().cast() };

    let reduced_x = quadrants_float.mul_add(
        Simd::splat(-PI2_C),
        quadrants_float.mul_add(Simd::splat(-PI2_B - PI2_A), abs_x),
    );

    (reduced_x, quadrants)
}

/// Calculates sine and cosine Taylor approximations of `x`. Doesn't perform any reductions, overflow
/// checking, etc.
#[inline]
fn sin_cos_taylor<const N: usize>(x: Simd<f32, N>) -> (Simd<f32, N>, Simd<f32, N>)
where
    LaneCount<N>: SupportedLaneCount,
{
    // These coefficients are not exactly the Taylor expansion but its approximation which
    // gives better accuracy on [-π/4, π/4].
    // src: https://github.com/vectorclass/version2/blob/master/vectormath_trig.h#L233-L239
    const P0_SIN: f32 = -1.666_665_5E-1;
    const P1_SIN: f32 = 8.332_161E-3;
    const P2_SIN: f32 = -1.951_529_6E-4;

    const P0_COS: f32 = 4.166_664_6E-2;
    const P1_COS: f32 = -1.388_731_6E-3;
    const P2_COS: f32 = 2.443_315_7E-5;

    let x2 = x * x;
    let sin = polynomial_simd!(x2; P0_SIN, P1_SIN, P2_SIN).mul_add(x2 * x, x);
    let cos = polynomial_simd!(x2; 1.0, -0.5, P0_COS, P1_COS, P2_COS);

    (sin, cos)
}

#[inline]
fn atan_taylor<const N: usize>(x: Simd<f32, N>) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    // src: https://github.com/vectorclass/version2/blob/master/vectormath_trig.h#L924-L927
    const P0: f32 = -3.333_295E-1;
    const P1: f32 = 1.997_771_1E-1;
    const P2: f32 = -1.387_768_5E-1;
    const P3: f32 = 8.053_744_6E-2;

    let x2 = x * x;
    polynomial_simd!(x2; P0, P1, P2, P3).mul_add(x2 * x, x)
}

impl<const N: usize> Trigonometry for Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
    Self: SimdFloat<Scalar = f32, Bits = Simd<u32, N>, Mask = Mask<i32, N>>,
{
    #[inline]
    fn sin(self) -> Self {
        let (reduced_x, quadrants) = trig_reduction(self);
        let (sin, cos) = sin_cos_taylor(reduced_x);

        let sin_cos_swap = (quadrants & Simd::splat(1)).simd_eq(Simd::default());
        let sin_vals = sin_cos_swap.select(sin, cos);
        sin_vals.sign_combine(Simd::from_bits(self.to_bits() ^ (quadrants << 30)))
    }

    #[inline]
    fn cos(self) -> Self {
        let (reduced_x, quadrants) = trig_reduction(self);
        let (sin, cos) = sin_cos_taylor(reduced_x);

        let sin_cos_swap = (quadrants & Simd::splat(1)).simd_eq(Simd::default());
        let cos_vals = sin_cos_swap.select(cos, sin);
        cos_vals.sign_combine(Simd::from_bits((quadrants + Simd::splat(1)) << 30))
    }

    #[inline]
    fn asin(self) -> Self {
        // src: https://github.com/vectorclass/version2/blob/master/vectormath_trig.h#L711-L715
        const P0: f32 = 1.666_675_2E-1;
        const P1: f32 = 7.495_300_3E-2;
        const P2: f32 = 4.547_002_6E-2;
        const P3: f32 = 2.418_131E-2;
        const P4: f32 = 4.216_32E-2;

        let x_abs = self.abs();
        let big = x_abs.simd_ge(Simd::splat(0.5));

        // for x >= 0: π/2 - 2 arcsin √((1-x)/2) = arcsin x
        // taylor_arg is less than or equal to 0.5
        let taylor_arg2 = big.select((Simd::splat(1.0) - x_abs) * Simd::splat(0.5), x_abs * x_abs);
        let taylor_arg = big.select(taylor_arg2.sqrt(), x_abs);
        let taylor_result = polynomial_simd!(taylor_arg2; P0, P1, P2, P3, P4)
            .mul_add(taylor_arg * taylor_arg2, taylor_arg);

        let asin_abs = big.select(
            Simd::splat(FRAC_PI_2) - (taylor_result + taylor_result),
            taylor_result,
        );
        asin_abs.sign_combine(self)
    }

    #[inline]
    fn acos(self) -> Self {
        Simd::splat(FRAC_PI_2) - self.asin()
    }

    #[inline]
    fn atan(self) -> Self {
        let abs_t = self.abs();
        // for |t| > √2 + 1 = tan(3π/8) ("big"): atan |t| = atan -1/|t| + π/2
        // for √2 - 1 = tan(π/8) <= |t| <= √2 + 1: atan |t| = atan (|t|-1)/(|t|+1) + π/4
        // for |t| < √2 - 1 ("small"): atan |t| = atan |t|/1
        let not_big = abs_t.simd_le(Simd::splat(SQRT_2 + 1.0));
        let not_small = abs_t.simd_ge(Simd::splat(SQRT_2 - 1.0));
        let reduced_arg = {
            let a = not_big.select(abs_t, Simd::default())
                - not_small.select(Simd::splat(1.0), Simd::default());
            let b = not_big.select(Simd::splat(1.0), Simd::default())
                + not_small.select(abs_t, Simd::default());
            a / b
        };
        let taylor_result = atan_taylor(reduced_arg);
        let atan_abs = taylor_result
            + not_small.select(
                not_big.select(Simd::splat(FRAC_PI_4), Simd::splat(FRAC_PI_2)),
                Simd::default(),
            );
        atan_abs.sign_combine(self)
    }

    #[inline]
    fn atan2(self, x: Self) -> Self {
        let abs_y = self.abs();
        let abs_x = x.abs();

        let not_big = (abs_x * Simd::splat(SQRT_2 + 1.0)).simd_ge(abs_y);
        let not_small = (abs_x * Simd::splat(SQRT_2 - 1.0)).simd_le(abs_y);
        let reduced_ratio = {
            let a =
                not_big.select(abs_y, Simd::default()) - not_small.select(abs_x, Simd::default());
            let b =
                not_big.select(abs_x, Simd::default()) + not_small.select(abs_y, Simd::default());
            (abs_y.is_finite() | abs_x.is_finite()).select(a / b, Simd::default())
        };
        let taylor_result = atan_taylor(reduced_ratio);
        let mut atan_abs = taylor_result
            + not_small.select(
                not_big.select(Simd::splat(FRAC_PI_4), Simd::splat(FRAC_PI_2)),
                Simd::default(),
            );
        // fix NaNs when x and y are both zeros
        atan_abs = Simd::from_bits(x.to_bits() | self.to_bits())
            .simd_eq(Simd::default())
            .select(Simd::from_bits(x.to_bits() ^ self.to_bits()), atan_abs);
        x.sign_bit()
            .simd_eq(Simd::default())
            .select(atan_abs, Simd::splat(PI) - atan_abs)
            .sign_combine(self)
    }
}
