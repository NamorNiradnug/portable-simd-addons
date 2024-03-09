use std::{
    f32::consts::FRAC_2_PI,
    simd::{
        cmp::{SimdPartialEq, SimdPartialOrd},
        num::SimdFloat,
        LaneCount, Simd, StdFloat, SupportedLaneCount,
    },
};

use crate::{math::util::sign_mask_f32, polynomial_simd};

pub trait Trigonometry {
    /// Calculates sine of each lane.
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
    fn cos(self) -> Self;

    #[inline]
    fn tan(self) -> Self
    where
        Self: std::ops::Div<Output = Self> + Copy,
    {
        self.sin() / self.cos()
    }

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
    fn atan2(self, other: Self) -> Self;
}

impl<const N: usize> Trigonometry for Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
    Self: SimdFloat<Scalar = f32, Cast<u32> = Simd<u32, N>, Bits = Simd<u32, N>>,
{
    fn sin(self) -> Self {
        // These coefficients are not exactly from Taylor expansion but its approximation
        const P0_SIN: f32 = -1.666_665_5E-1;
        const P1_SIN: f32 = 8.332_161E-3;
        const P2_SIN: f32 = -1.951_529_6E-4;

        const P0_COS: f32 = 4.166_664_6E-2;
        const P1_COS: f32 = -1.388_731_6E-3;
        const P2_COS: f32 = 2.443_315_7E-5;

        // XXX: What're those numbers??
        const DP1: f32 = 0.78515625 * 2.0;
        const DP2: f32 = 2.418_756_5E-4 * 2.0;
        const DP3: f32 = 3.774_895E-8 * 2.0;

        const INPUT_LIMIT: f32 = 1e5;

        let mut abs_values = self.abs();
        // NaNs and large values are mapped to 0
        abs_values = abs_values
            .simd_lt(Simd::splat(INPUT_LIMIT))
            .select(abs_values, Simd::default());
        let quadrants_float = (abs_values * Simd::splat(FRAC_2_PI)).round();

        // SAFETY: values in `quadrants_float` are finite and between 0 and 1e5
        let quadrants = unsafe { quadrants_float.to_int_unchecked::<u32>() };

        // XXX: what's going on here??
        let reduced_x = quadrants_float.mul_add(
            Simd::splat(-DP3),
            quadrants_float.mul_add(
                Simd::splat(-DP2),
                quadrants_float.mul_add(Simd::splat(-DP1), abs_values),
            ),
        );

        let reduced_x2 = reduced_x * reduced_x;
        let sin = polynomial_simd!(reduced_x2; P0_SIN, P1_SIN, P2_SIN)
            .mul_add(reduced_x2 * reduced_x, reduced_x);
        let cos = polynomial_simd!(reduced_x2; 1.0, -0.5, P0_COS, P1_COS, P2_COS);

        let sin_cos_swap = (quadrants & Simd::splat(1)).simd_ne(Simd::default());
        let sin_vals = sin_cos_swap.select(cos, sin);
        Simd::from_bits(
            sin_vals.to_bits() ^ ((quadrants & Simd::splat(2)) << 30) ^ sign_mask_f32(self),
        )
    }

    fn cos(self) -> Self {
        todo!()
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

    fn atan2(self, _other: Self) -> Self {
        todo!()
    }
}
