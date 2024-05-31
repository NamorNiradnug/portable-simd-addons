#[cfg(all(not(target_arch = "x86"), not(target_arch = "x86_64")))]
compile_error!("VCL supports only x86-compatible platforms");

#[cxx::bridge(namespace = "bench")]
pub mod ffi {
    unsafe extern "C++" {
        include!("portable-simd-addons/benches/cpp/vclbench.hpp");

        unsafe fn exp_f32_vcl(x: *const f32, result: *mut f32);
        unsafe fn exp2_f32_vcl(x: *const f32, result: *mut f32);
        unsafe fn exp_m1_f32_vcl(x: *const f32, result: *mut f32);

        unsafe fn sin_f32_vcl(x: *const f32, result: *mut f32);
        unsafe fn cos_f32_vcl(x: *const f32, result: *mut f32);
        unsafe fn tan_f32_vcl(x: *const f32, result: *mut f32);

        unsafe fn asin_f32_vcl(x: *const f32, result: *mut f32);
        unsafe fn acos_f32_vcl(x: *const f32, result: *mut f32);
        unsafe fn atan_f32_vcl(x: *const f32, result: *mut f32);

        unsafe fn exp_f64_vcl(x: *const f64, result: *mut f64);
        unsafe fn exp2_f64_vcl(x: *const f64, result: *mut f64);
        unsafe fn exp_m1_f64_vcl(x: *const f64, result: *mut f64);

        unsafe fn sin_f64_vcl(x: *const f64, result: *mut f64);
        unsafe fn cos_f64_vcl(x: *const f64, result: *mut f64);
        unsafe fn tan_f64_vcl(x: *const f64, result: *mut f64);

        unsafe fn asin_f64_vcl(x: *const f64, result: *mut f64);
        unsafe fn acos_f64_vcl(x: *const f64, result: *mut f64);
        unsafe fn atan_f64_vcl(x: *const f64, result: *mut f64);
    }
}

pub use ffi::*;
