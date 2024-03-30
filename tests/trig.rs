#![feature(portable_simd)]

mod common;
use std::f32::INFINITY;

use common::Linspace;

use simd_addons::math::Trigonometry;

approx_test_simd_fn!(
    f32 {
        sin: (-1e3..1e3).linspace(100_000);
        cos: (-1e3..1e3).linspace(100_000);
        tan: (-1e3..1e3).linspace(100_000);
        asin: (-1.0..1.0).linspace(100_000);
        acos: (-1.0..1.0).linspace(100_000);
        atan: (-1e3..1e3).linspace(100_000), [INFINITY];
    }
    f64 {
        sin: (-1e5..1e5).linspace(1_000_000);
        cos: (-1e5..1e5).linspace(1_000_000);
        tan: (-1e5..1e5).linspace(1_000_000);
    }
);
