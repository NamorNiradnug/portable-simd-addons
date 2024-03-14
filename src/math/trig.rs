use std::{
    f32::consts::{FRAC_2_PI, FRAC_PI_2},
    simd::{
        cmp::{SimdPartialEq, SimdPartialOrd},
        num::SimdFloat,
        LaneCount, Simd, StdFloat, SupportedLaneCount,
    },
};

use crate::{math::util::FloatBitUtils, polynomial_simd};

pub trait Trigonometry {
    /// Calculates sine of each lane. Overflows on large values (greater than 10⁵) and return `0.0`
    /// in the case.
    ///
    /// # Examples
    /// ```
    /// # #![feature(portable_simd)]
    /// # use simd_addons::math::*;
    /// # use std::simd::prelude::*;
    /// let values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
    /// let vec_sin = Simd::from(values).sin();
    /// let loop_sin = Simd::from(values.map(|x| f32::sin(x)));
    /// assert!((vec_sin - loop_sin).abs().reduce_max() < 1e-7);
    /// ```
    fn sin(self) -> Self;

    /// Calculates cosine of each lane. Overflows on large values (greater than 10⁵) and returns
    /// `1.0` in the case.
    fn cos(self) -> Self;

    #[inline]
    fn tan(self) -> Self
    where
        Self: std::ops::Div<Output = Self> + Copy,
    {
        let (sin, cos) = self.sin_cos();
        sin / cos
    }

    /// Calculates both sine and cosine of each lane.
    /// Overflows on large values (see [`cos`] and [`sin`] documentation for details).
    ///
    /// [`cos`]: `Self::cos`
    /// [`sin`]: `Self::sin`
    ///
    /// # Examples
    /// ```
    /// # #![feature(portable_simd)]
    /// # use std::simd::prelude::f32x4;
    /// # use simd_addons::math::*;
    /// let x = f32x4::from_array([0.0, 1.0, 2.0, 3.0]);
    /// assert_eq!(x.sin_cos(), (x.sin(), x.cos()));
    /// ```
    #[inline]
    fn sin_cos(self) -> (Self, Self)
    where
        Self: Copy,
    {
        (self.sin(), self.cos())
    }

    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan(self) -> Self;
    fn atan2(self, x: Self) -> Self;
}

/// π/4 reduction of `x`. Large values (greater than 10⁵) are treated as zeros.
#[inline]
fn trig_reduction_f32<const N: usize>(x: Simd<f32, N>) -> (Simd<f32, N>, Simd<u32, N>)
where
    LaneCount<N>: SupportedLaneCount,
{
    const INPUT_LIMIT: f32 = 1e5;

    const PI2_A: f32 = 0.785_156_25 * 2.0;
    const PI2_B: f32 = 2.418_756_5E-4 * 2.0;
    const PI2_C: f32 = 3.774_895E-8 * 2.0;

    let mut abs_x = x.abs();
    // NaNs, INFs and large values are mapped to 0
    abs_x = abs_x
        .simd_lt(Simd::splat(INPUT_LIMIT))
        .select(abs_x, Simd::default());
    let quadrants_float = (abs_x * Simd::splat(FRAC_2_PI)).round();

    // SAFETY: values in `quadrants_float` are finite and between 0 and 1e5
    let quadrants = unsafe { quadrants_float.to_int_unchecked::<u32>() };

    let reduced_x = quadrants_float.mul_add(
        Simd::splat(-PI2_C),
        quadrants_float.mul_add(
            Simd::splat(-PI2_B),
            quadrants_float.mul_add(Simd::splat(-PI2_A), abs_x),
        ),
    );

    (reduced_x, quadrants)
}

/// Calculates sine of and cosine Taylor approximations of `x`. Doesn't perform any reductions, overflow
/// checking, etc.
#[inline]
fn sin_cos_taylor_f32<const N: usize>(x: Simd<f32, N>) -> (Simd<f32, N>, Simd<f32, N>)
where
    LaneCount<N>: SupportedLaneCount,
{
    // These coefficients are not exactly the Taylor expansion but its approximation which
    // gives better accuracy on [-π/4, π/4].
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
fn sin_sign_adj<const N: usize>(quadrants: Simd<u32, N>) -> Simd<u32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    (quadrants & Simd::splat(2)) << 30
}

#[inline]
fn cos_sign_adj<const N: usize>(quadrants: Simd<u32, N>) -> Simd<u32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    ((quadrants + Simd::splat(1)) & Simd::splat(2)) << 30
}

impl<const N: usize> Trigonometry for Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
    Self: SimdFloat<Scalar = f32, Bits = Simd<u32, N>>,
{
    #[inline]
    fn sin(self) -> Self {
        let (reduced_x, quadrants) = trig_reduction_f32(self);
        let (sin, cos) = sin_cos_taylor_f32(reduced_x);

        let sin_cos_swap = (quadrants & Simd::splat(1)).simd_eq(Simd::default());
        let sin_vals = sin_cos_swap.select(sin, cos);
        sin_vals
            .sign_combine(self)
            .sign_combine(Simd::from_bits(sin_sign_adj(quadrants)))
    }

    #[inline]
    fn cos(self) -> Self {
        let (reduced_x, quadrants) = trig_reduction_f32(self);
        let (sin, cos) = sin_cos_taylor_f32(reduced_x);

        let sin_cos_swap = (quadrants & Simd::splat(1)).simd_eq(Simd::default());
        let cos_vals = sin_cos_swap.select(cos, sin);
        cos_vals.sign_combine(Simd::from_bits(cos_sign_adj(quadrants)))
    }

    #[inline]
    fn asin(self) -> Self {
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

    fn acos(self) -> Self {
        Simd::splat(FRAC_PI_2) - self.asin()
    }

    fn atan(self) -> Self {
        todo!()
    }

    fn atan2(self, _other: Self) -> Self {
        todo!()
    }
}

#[cfg(test)]
mod test {
    extern crate test;
    use crate::math::Trigonometry;
    use approx::*;
    use std::ops::Range;
    use std::simd::prelude::*;

    fn linspace(range: Range<f32>, n: usize) -> Vec<f32> {
        Vec::from_iter(
            (0..=n).map(|i| range.start + (range.end - range.start) * (i as f32) / (n as f32)),
        )
    }

    const BENCH_POINTS: usize = 200_000;

    #[bench]
    #[no_mangle]
    fn vec_sin_bench(b: &mut test::Bencher) {
        let data = linspace(-1e4..1e4, BENCH_POINTS);
        b.iter(|| {
            data.array_chunks::<64>()
                .map(|x| Simd::from_array(*x).sin())
                .sum::<f32x64>()
        })
    }

    #[bench]
    fn scalar_sin_bench(b: &mut test::Bencher) {
        let data = linspace(-1e4..1e4, BENCH_POINTS);
        b.iter(|| data.iter().map(|x| x.sin()).sum::<f32>())
    }

    #[bench]
    fn vec_sin_cos_bench(b: &mut test::Bencher) {
        let data = linspace(-1e4..1e4, BENCH_POINTS);
        b.iter(|| {
            data.array_chunks::<64>()
                .map(|x| {
                    let (sin, cos) = Simd::from_array(*x).sin_cos();
                    sin + cos
                })
                .sum::<f32x64>()
        })
    }

    #[test]
    fn sin_accurancy() {
        let data = linspace(-1e4..1e4, 1_000_000);
        for x in data {
            assert_abs_diff_eq!(x.sin(), f32x1::splat(x).sin()[0], epsilon = 10e-6);
        }
    }

    #[test]
    fn cos_accurancy() {
        let data = linspace(-1e4..1e4, 1_000_000);
        for x in data {
            assert_abs_diff_eq!(x.cos(), f32x1::splat(x).cos()[0], epsilon = 10e-6);
        }
    }

    #[test]
    fn asin_acos_accurancy() {
        let data = linspace(-1.0..1.0, 100_000);
        for x in data {
            let vec = f32x1::splat(x);
            assert_ulps_eq!(x.asin(), vec.asin()[0]);
            assert_ulps_eq!(x.acos(), vec.acos()[0]);
            assert_ulps_eq!(x, vec.asin().sin()[0]);
        }
    }
}
