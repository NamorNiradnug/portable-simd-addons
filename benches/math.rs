#![feature(portable_simd, test, array_chunks, core_intrinsics)]
#![allow(internal_features)]
extern crate test;

#[path = "../tests/common/mod.rs"]
mod common;
use common::Linspace;

use simd_addons::math::{Exponent, Trigonometry};
use std::simd::prelude::*;

const BENCH_POINTS: usize = 200_000;

#[cfg(feature = "vectorclass_bench")]
#[path = "cpp/vclbench.rs"]
mod vclbench;

#[cfg(feature = "libmvec_bench")]
#[path = "cpp/libmvecbench.rs"]
mod libmvecbench;

macro_rules! bench_func {
    ($range: expr, $func: tt, $ftype: ty) => {
        bench_func!($range, $func, $ftype, 64);
    };

    ($range: expr, $func: tt, $ftype: ty, $vecsize: literal) => {
        paste::paste! {
        #[bench]
        fn [< bench_ $func _ $ftype _vec >](b: &mut test::Bencher) {
            #[allow(clippy::all)]
            let data: Vec<_> = ($range as $ftype).linspace(BENCH_POINTS).collect();
            let x = data.as_slice();
            let mut result_vec: Vec<_> = vec![0.0; BENCH_POINTS];
            let result = result_vec.as_mut_slice();
            b.iter(|| {
                assert_eq!(x.len(), BENCH_POINTS);
                assert_eq!(result.len(), BENCH_POINTS);
                for i in (0..BENCH_POINTS).step_by($vecsize) {
                    Simd::<_, $vecsize>::from_slice(&x[i..])
                        .$func()
                        .copy_to_slice(&mut result[i..]);
                }
            })
        }

        #[bench]
        fn [<bench_ $func _ $ftype _scalar >](b: &mut test::Bencher) {
            #[allow(clippy::all)]
            let data: Vec<_> = ($range as $ftype).linspace(BENCH_POINTS).collect();
            let mut result = vec![0.0; BENCH_POINTS];
            b.iter(|| {
                assert_eq!(result.len(), BENCH_POINTS);
                assert_eq!(data.len(), BENCH_POINTS);
                for (x, res) in std::iter::zip(&data, &mut result) {
                    *res = x.$func();
                }
            });
        }


        #[cfg(feature = "vectorclass_bench")]
        #[bench]
        fn [<bench_ $func _ $ftype _vcl >](b: &mut test::Bencher) {
            let data: Vec<$ftype> = ($range as $ftype).linspace(BENCH_POINTS).collect();
            let mut result = vec![0.0; BENCH_POINTS];
            b.iter(|| {
                unsafe { vclbench::[<$func _ $ftype _vcl>](data.as_ptr(), result.as_mut_ptr()) }
            });
        }

        #[cfg(feature = "libmvec_bench")]
        #[bench]
        fn [<bench_ $func _ $ftype _libmvec >](b: &mut test::Bencher) {
            let data: Vec<$ftype> = ($range as $ftype).linspace(BENCH_POINTS).collect();
            let mut result = vec![0.0; BENCH_POINTS];
            b.iter(|| {
                unsafe { libmvecbench::[<$func _ $ftype _libmvec>](data.as_ptr(), result.as_mut_ptr()) }
            });
        }
        }
    };
}

bench_func!(-50.0..50, exp, f32, 16);
bench_func!(-50.0..50, exp, f64);
bench_func!(-50.0..50, exp2, f32, 16);
bench_func!(-50.0..50, exp2, f64);
bench_func!(-50.0..50, exp_m1, f32, 16);
bench_func!(-50.0..50, exp_m1, f64);

bench_func!(-1e4..1e4, sin, f32);
bench_func!(-1e4..1e4, sin, f64);
bench_func!(-1e4..1e4, cos, f32);
bench_func!(-1e4..1e4, cos, f64);
bench_func!(-1e4..1e4, tan, f32);
bench_func!(-1e4..1e4, tan, f64);

bench_func!(-1.0..1.0, asin, f32);
bench_func!(-1.0..1.0, asin, f64);
bench_func!(-1.0..1.0, acos, f32);
bench_func!(-1.0..1.0, acos, f64);
bench_func!(-1e3..1e3, atan, f32);
bench_func!(-1e3..1e3, atan, f64);

#[allow(unused)]
fn generate_atan2_bench_sample() -> (Vec<f32>, Vec<f32>) {
    todo!()
}
