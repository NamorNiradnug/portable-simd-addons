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
