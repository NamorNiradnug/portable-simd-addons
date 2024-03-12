/// Evaluates polynomial with argument of [`Simd`](std::simd::Simd) type using [Estrin's scheme][scheme_wiki].
///
/// [scheme_wiki]: https://en.wikipedia.org/wiki/Estrin's_scheme
///
/// # Examples
/// ```
/// # #![feature(portable_simd)]
/// # use std::simd::{prelude::*, *};
/// # use simd_addons::polynomial_simd;
/// let x = f32x4::from_array([-1.0, 0.0, 1.0, 2.0]);
/// assert_eq!(
///     // p(x) = 1 + 2x + 3x²
///     polynomial_simd!(x; 1.0, 2.0, 3.0),
///     Simd::splat(1.0) + Simd::splat(2.0) * x + Simd::splat(3.0) * x * x
/// );
/// assert_eq!(
///     // p(x) = 1 - 2x + x³
///     polynomial_simd!(x; 1.0, -2.0, 0.0, 1.0),
///     Simd::splat(1.0) - Simd::splat(2.0) * x + x * x * x
/// );
/// ```
#[macro_export]
macro_rules! polynomial_simd {
    ($x: expr; $coef_0: expr) => {
        Simd::splat($coef_0)
    };

    ($x: expr; $coef_0: expr, $coef_1: expr) => {
        $x.mul_add(Simd::splat($coef_1), Simd::splat($coef_0))
    };

    ($x: expr; $( $coef_even: expr, $coef_odd: expr ),+) => {{
        let x = $x;
        let x2 = x * x;
        x.mul_add(polynomial_simd!(x2; $( $coef_odd ),*), polynomial_simd!(x2; $( $coef_even ),*))
    }};

    ($x: expr; $coef_0: expr, $( $coef_odd: expr, $coef_even: expr ),+) => {{
        let x = $x;
        let x2 = x * x;
        x.mul_add(
            polynomial_simd!(x2; $( $coef_odd ),*),
            polynomial_simd!(x2; $coef_0, $( $coef_even ),*),
        )
    }};
}
