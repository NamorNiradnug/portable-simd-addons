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
mod cpp_benches {
    #[cfg(all(not(target_arch = "x86"), not(target_arch = "x86_64")))]
    compile_error!("VCL supports only x86-compatible platforms");

    use super::{Linspace, BENCH_POINTS};

    #[cxx::bridge(namespace = "bench")]
    mod ffi {
        unsafe extern "C++" {
            include!("portable-simd-addons/benches/vclbench.hpp");

            unsafe fn exp_f32_vcl(x: *const f32, result: *mut f32);
            unsafe fn exp2_f32_vcl(x: *const f32, result: *mut f32);
            unsafe fn expm1_f32_vcl(x: *const f32, result: *mut f32);

            unsafe fn sin_f32_vcl(x: *const f32, result: *mut f32);
            unsafe fn cos_f32_vcl(x: *const f32, result: *mut f32);
            unsafe fn tan_f32_vcl(x: *const f32, result: *mut f32);

            unsafe fn asin_f32_vcl(x: *const f32, result: *mut f32);
            unsafe fn acos_f32_vcl(x: *const f32, result: *mut f32);
            unsafe fn atan_f32_vcl(x: *const f32, result: *mut f32);

            unsafe fn exp_f32_scalar(x: *const f32, result: *mut f32);
            unsafe fn exp2_f32_scalar(x: *const f32, result: *mut f32);
            unsafe fn expm1_f32_scalar(x: *const f32, result: *mut f32);

            unsafe fn sin_f32_scalar(x: *const f32, result: *mut f32);
            unsafe fn cos_f32_scalar(x: *const f32, result: *mut f32);
            unsafe fn tan_f32_scalar(x: *const f32, result: *mut f32);

            unsafe fn asin_f32_scalar(x: *const f32, result: *mut f32);
            unsafe fn acos_f32_scalar(x: *const f32, result: *mut f32);
            unsafe fn atan_f32_scalar(x: *const f32, result: *mut f32);
        }
    }
    macro_rules! bench_cpp {
        ($($func: tt, $range: expr);* $(;)?) => {
            $(
            paste::paste! {
                #[allow(non_snake_case)]
                #[bench]
                fn [< bench_ $func _f32_scalar_cpp >](b: &mut test::Bencher) {
                    let data: Vec<f32> = ($range).linspace(BENCH_POINTS).collect();
                    let mut result = vec![0.0; BENCH_POINTS];
                    b.iter(|| {
                        unsafe { ffi::[<$func _f32_scalar>](data.as_ptr(), result.as_mut_ptr()) }
                    });
                }

                #[allow(non_snake_case)]
                #[bench]
                fn [< bench_ $func _f32_vcl >](b: &mut test::Bencher) {
                    let data: Vec<f32> = ($range).linspace(BENCH_POINTS).collect();
                    let mut result = vec![0.0; BENCH_POINTS];
                    b.iter(|| {
                        unsafe { ffi::[<$func _f32_vcl>](data.as_ptr(), result.as_mut_ptr()) }
                    });
                }
            }
            )*
        };
    }

    bench_cpp!(
        exp, -50.0..50.0f32;
        exp2, -50.0..50.0f32;
        expm1, -50.0..50.0f32;
        sin, -1e4..1e4f32;
        cos, -1e4..1e4f32;
        tan, -1e4..1e4f32;
        asin, -1.0..1.0f32;
        acos, -1.0..1.0f32;
        atan, -1e3..1e3f32;
    );
}

macro_rules! bench_simd_vs_scalar {
    ($range: expr, $func: tt, $ftype: ty) => {
        bench_simd_vs_scalar!($range, $func, $ftype, 64);
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
        fn [< bench_ $func _ $ftype _scalar >](b: &mut test::Bencher) {
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
        }
    };
}

bench_simd_vs_scalar!(-50.0..50, exp, f32, 16);
bench_simd_vs_scalar!(-50.0..50, exp, f64);
bench_simd_vs_scalar!(-50.0..50, exp2, f32, 16);
bench_simd_vs_scalar!(-50.0..50, exp2, f64);
bench_simd_vs_scalar!(-50.0..50, exp_m1, f32, 16);
bench_simd_vs_scalar!(-50.0..50, exp_m1, f64);

bench_simd_vs_scalar!(-1e4..1e4, sin, f32);
bench_simd_vs_scalar!(-1e4..1e4, sin, f64);
bench_simd_vs_scalar!(-1e4..1e4, cos, f32);
bench_simd_vs_scalar!(-1e4..1e4, cos, f64);
bench_simd_vs_scalar!(-1e4..1e4, tan, f32);
bench_simd_vs_scalar!(-1e4..1e4, tan, f64);

bench_simd_vs_scalar!(-1.0..1.0, asin, f32);
bench_simd_vs_scalar!(-1.0..1.0, asin, f64);
bench_simd_vs_scalar!(-1.0..1.0, acos, f32);
bench_simd_vs_scalar!(-1.0..1.0, acos, f64);
bench_simd_vs_scalar!(-1e3..1e3, atan, f32);
bench_simd_vs_scalar!(-1e3..1e3, atan, f64);

#[allow(unused)]
fn generate_atan2_bench_sample() -> (Vec<f32>, Vec<f32>) {
    todo!()
}
