#include "vclbench.hpp"

#define VCL_NAMESPACE vcl
#include "vcl/vectormath_exp.h"
#include "vcl/vectormath_trig.h"

static constexpr size_t BENCH_POINTS = 200'000;

using Vec16fFunc = vcl::Vec16f (*)(vcl::Vec16f);

template <Vec16fFunc func>
void vcl_bench_f32_impl(const float *x, float *result) {
    vcl::Vec16f x_vec;
    for (size_t i = 0; i < BENCH_POINTS; i += vcl::Vec16f::size()) {
        x_vec.load(x + i);
        func(x_vec).store(result + i);
    }
}

namespace bench {
void exp_f32_vcl(const float *x, float *result) { vcl_bench_f32_impl<vcl::exp>(x, result); }
void sin_f32_vcl(const float *x, float *result) { vcl_bench_f32_impl<vcl::sin>(x, result); }
void asin_f32_vcl(const float *x, float *result) { vcl_bench_f32_impl<vcl::asin>(x, result); }
void atan_f32_vcl(const float *x, float *result) { vcl_bench_f32_impl<vcl::atan>(x, result); }

void atan2_f32_vcl(const float *x, const float *y, float *result) {
    vcl::Vec16f x_vec;
    vcl::Vec16f y_vec;
    for (size_t i = 0; i < BENCH_POINTS; i += vcl::Vec16f::size()) {
        x_vec.load(x + i);
        y_vec.load(y + i);
        vcl::atan2(y_vec, x_vec).store(result + i);
    }
}

void exp_f32_scalar(const float *x, float *result) {
    for (size_t i = 0; i < BENCH_POINTS; ++i) {
        result[i] = expf(x[i]);
    }
}

void sin_f32_scalar(const float *x, float *result) {
    for (size_t i = 0; i < BENCH_POINTS; ++i) {
        result[i] = sinf(x[i]);
    }
}
}  // namespace bench
