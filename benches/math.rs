#![feature(portable_simd, test, array_chunks, core_intrinsics)]
#![allow(internal_features)]
extern crate test;

#[path = "../tests/common/mod.rs"]
mod common;
use common::Linspace;

use simd_addons::math::Trigonometry;
use std::simd::prelude::*;

const BENCH_POINTS: usize = 200_000;

#[cfg(feature = "vectorclass_bench")]
mod vcl_bench {
    use super::Linspace;
    use super::BENCH_POINTS;

    #[cfg(all(not(target_arch = "x86"), not(target_arch = "x86_64")))]
    compile_error!("VCL supports only x86-compatible platforms");

    #[cxx::bridge(namespace = "bench")]
    mod ffi {
        unsafe extern "C++" {
            include!("portable-simd-addons/benches/vclbench.hpp");
            unsafe fn ExpVCL(x: *const f32, result: *mut f32);
            unsafe fn SinVCL(x: *const f32, result: *mut f32);
            unsafe fn ExpScalar(x: *const f32, result: *mut f32);
            unsafe fn SinScalar(x: *const f32, result: *mut f32);
        }
    }

    macro_rules! bench_cpp {
        ($range: expr, $func: tt) => {
            paste::paste! {
            #[allow(non_snake_case)]
            #[bench]
            fn [< bench_ $func _cpp >](b: &mut test::Bencher) {
                let data: Vec<_> = ($range).linspace(BENCH_POINTS).collect();
                let mut result = vec![0.0; BENCH_POINTS];
                b.iter(|| {
                    unsafe { ffi::$func(data.as_ptr(), result.as_mut_ptr()) }
                });
            }
            }
        };
    }

    bench_cpp!(-1e4..1e4f32, SinVCL);
    bench_cpp!(-1e4..1e4f32, SinScalar);
    bench_cpp!(-50.0..50.0f32, ExpVCL);
    bench_cpp!(-50.0..50.0f32, ExpScalar);
}

macro_rules! bench_simd_vs_scalar {
    ($range: expr, $func: tt, $ftype: ty $(, $coresimdfn: tt )?) => {
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
                for i in (0..BENCH_POINTS).step_by(64) {
                    Simd::<_, 64>::from_slice(&x[i..])
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

        $(
        #[bench]
        fn [< bench_ $func _ $ftype _core_simd >](b: &mut test::Bencher) {
            #[allow(clippy::all)]
            let data: Vec<_> = ($range as $ftype).linspace(BENCH_POINTS).collect();
            let x = data.as_slice();
            let mut result_vec: Vec<_> = vec![0.0; BENCH_POINTS];
            let result = result_vec.as_mut_slice();
            b.iter(|| {
                assert_eq!(x.len(), BENCH_POINTS);
                assert_eq!(result.len(), BENCH_POINTS);
                for i in (0..BENCH_POINTS).step_by(64) {
                    unsafe { core::intrinsics::simd::$coresimdfn(
                        Simd::<_, 64>::from_slice(&x[i..]))
                    }
                    .copy_to_slice(&mut result[i..]);
                }
            });
        }
        )?
        }
    };
}

bench_simd_vs_scalar!(-1e4..1e4, sin, f32, simd_fsin);
bench_simd_vs_scalar!(-1e4..1e4, sin, f64, simd_fsin);
bench_simd_vs_scalar!(-1e4..1e4, cos, f32, simd_fcos);
bench_simd_vs_scalar!(-1e4..1e4, cos, f64, simd_fcos);
