#![feature(portable_simd)]

mod common;

use common::Linspace;

use simd_addons::math::{Exponent, Trigonometry};

approx_test_simd_fn!(
    f32 {
        sin: (-1e3..1e3).linspace(100_000);
        cos: (-1e3..1e3).linspace(100_000);
        tan: (-1e3..1e3).linspace(100_000);
        asin: (-1.0..1.0).linspace(100_000);
        acos: (-1.0..1.0).linspace(100_000);
        atan: (-1e3..1e3).linspace(100_000), [f32::INFINITY, -f32::INFINITY];

        exp: (-88.0..88.0).linspace(100_000), [1e3, -1e3, f32::INFINITY, -f32::INFINITY];
        exp_m1: (-88.0..88.0).linspace(100_000), [1e3, -1e3, f32::INFINITY, -f32::INFINITY];
        exp2: (-127.0..127.0).linspace(100_000), [1e3, -1e3, f32::INFINITY, -f32::INFINITY];
    }
    f64 {
        sin: (-1e5..1e5).linspace(1_000_000);
        cos: (-1e5..1e5).linspace(1_000_000);
        tan: (-1e5..1e5).linspace(1_000_000);
        asin: (-1.0..1.0).linspace(100_000);
        acos: (-1.0..1.0).linspace(100_000);
        atan: (-1e3..1e3).linspace(100_000), [f64::INFINITY, -f64::INFINITY];
    }
);

#[test]
fn test_simd_atan2_f32() {
    const VALUES: [f32; 6] = [0.0, -0.0, 1.0, -1.0, f32::INFINITY, -f32::INFINITY];
    for y in VALUES {
        for x in VALUES {
            assert_eq!(y.atan2(x), simd_fn!(y.atan2(x)), "atan2({}, {})", x, y);
        }
    }

    for x in (-10.0..10.0f32).linspace(1_000) {
        for y in (-10.0..10.0f32).linspace(1_000) {
            approx::assert_ulps_eq!(y.atan2(x), simd_fn!(y.atan2(x)));
        }
    }
}

#[test]
fn test_simd_atan2_f64() {
    const VALUES: [f64; 6] = [0.0, -0.0, 1.0, -1.0, f64::INFINITY, -f64::INFINITY];
    for y in VALUES {
        for x in VALUES {
            assert_eq!(y.atan2(x), simd_fn!(y.atan2(x)), "atan2({}, {})", x, y);
        }
    }

    for x in (-10.0..10.0f64).linspace(1_000) {
        for y in (-10.0..10.0f64).linspace(1_000) {
            approx::assert_ulps_eq!(y.atan2(x), simd_fn!(y.atan2(x)));
        }
    }
}
