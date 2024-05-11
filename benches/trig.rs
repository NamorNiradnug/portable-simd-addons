#![feature(portable_simd, test, array_chunks)]
extern crate test;

#[path = "../tests/common/mod.rs"]
mod common;
use common::Linspace;

use simd_addons::math::{Exponent, Trigonometry};
use std::simd::prelude::*;

const BENCH_POINTS: usize = 200_000;

#[bench]
fn vec_sin_bench(b: &mut test::Bencher) {
    let data: Vec<_> = (-1e4..1e4f32).linspace(BENCH_POINTS).collect();
    b.iter(|| {
        for x in test::black_box(data.array_chunks::<64>()) {
            test::black_box(Simd::from_array(*x).sin());
        }
    })
}

#[bench]
fn scalar_sin_bench(b: &mut test::Bencher) {
    let data: Vec<_> = (-1e4..1e4f32).linspace(BENCH_POINTS).collect();
    b.iter(|| {
        for x in &data {
            test::black_box(x.sin());
        }
    });
}

#[bench]
fn vec_sin_cos_bench(b: &mut test::Bencher) {
    let data: Vec<_> = (-1e4..1e4f32).linspace(BENCH_POINTS).collect();
    b.iter(|| {
        for x in test::black_box(data.array_chunks::<64>()) {
            test::black_box(Simd::from_array(*x).sin_cos());
        }
    })
}

#[bench]
fn vec_exp_bench(b: &mut test::Bencher) {
    let data: Vec<_> = (-1e2..1e2f32).linspace(BENCH_POINTS).collect();
    b.iter(|| {
        for x in test::black_box(data.array_chunks::<64>()) {
            test::black_box(Simd::from_array(*x).exp());
        }
    })
}

#[bench]
fn scalar_exp_bench(b: &mut test::Bencher) {
    let data: Vec<_> = (-1e2..1e2f32).linspace(BENCH_POINTS).collect();
    b.iter(|| {
        for x in &data {
            test::black_box(x.exp());
        }
    });
}
