use std::ops::Range;

pub trait Linspace {
    fn linspace(&self, n: usize) -> impl Iterator;
}

impl Linspace for Range<f32> {
    #[allow(refining_impl_trait)]
    fn linspace(&self, n: usize) -> impl Iterator<Item = f32> {
        (0..=n).map(move |i| self.start + (self.end - self.start) * (i as f32) / (n as f32))
    }
}

impl Linspace for Range<f64> {
    #[allow(refining_impl_trait)]
    fn linspace(&self, n: usize) -> impl Iterator<Item = f64> {
        (0..=n).map(move |i| self.start + (self.end - self.start) * (i as f64) / (n as f64))
    }
}

#[macro_export]
macro_rules! simd_fn {
    ($x: tt $( .$func: tt ( $( $args: tt ),* )) *) => {
        std::simd::Simd::<_, 1>::splat($x)$( .$func( $( std::simd::Simd::splat($args) ),* ) )*[0]
    };
}

#[macro_export]
macro_rules! approx_test_simd_fn {
    ( $( $ftype: ty { $( $fn: tt : $( $values: expr ),+ );* $(;)? } )+ ) => {
        $(
        $(
        paste::paste! {
        #[test]
        fn [< test_simd_ $fn _ $ftype >]() {
            $(
            for x in $values {
                approx::assert_ulps_eq!(
                    simd_fn!((x as $ftype).$fn()),
                    (x as $ftype).$fn(),
                    max_ulps = 5,
                )
            }
            )*
        }
        }
        )*
        )*
    };

}
