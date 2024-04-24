#![feature(portable_simd, test, array_chunks, core_intrinsics)]
#![allow(internal_features)]
extern crate test;

#[path = "../tests/common/mod.rs"]
mod common;
use common::Linspace;

use simd_addons::math::Trigonometry;
use std::simd::prelude::*;

const BENCH_POINTS: usize = 200_000;

macro_rules! bench_simd_vs_scalar {
    ($range: expr, $func: tt, $ftype: ty $(, $coresimdfn: tt )?) => {
        paste::paste! {
        #[bench]
        fn [< bench_ $func _ $ftype _vec >](b: &mut test::Bencher) {
            #[allow(clippy::all)]
            let data: Vec<_> = ($range as $ftype).linspace(BENCH_POINTS).collect();
            b.iter(|| {
                for x in data.array_chunks::<64>() {
                    test::black_box(Simd::from_array(*x).$func());
                }
            })
        }

        #[bench]
        fn [< bench_ $func _ $ftype _scalar >](b: &mut test::Bencher) {
            #[allow(clippy::all)]
            let data: Vec<_> = ($range as $ftype).linspace(BENCH_POINTS).collect();
            b.iter(|| {
                for x in &data {
                    test::black_box(x.$func());
                }
            });
        }

        $(
        #[bench]
        fn [< bench_ $func _ $ftype _core_simd >](b: &mut test::Bencher) {
            #[allow(clippy::all)]
            let data: Vec<_> = ($range as $ftype).linspace(BENCH_POINTS).collect();
            b.iter(|| {
                for x in data.array_chunks::<64>() {
                    test::black_box(unsafe { core::intrinsics::simd::$coresimdfn(Simd::from_array(*x)) });
                }
            })
        }
        )?
        }
    };
}

bench_simd_vs_scalar!(-1e4..1e4, sin, f32, simd_fsin);
bench_simd_vs_scalar!(-1e4..1e4, sin, f64, simd_fsin);
bench_simd_vs_scalar!(-1e4..1e4, cos, f32, simd_fcos);
bench_simd_vs_scalar!(-1e4..1e4, cos, f64, simd_fcos);
