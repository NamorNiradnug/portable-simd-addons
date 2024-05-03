#include "vclbench.hpp"

#define VCL_NAMESPACE vcl
#include "vcl/vectormath_exp.h"
#include "vcl/vectormath_trig.h"

static constexpr size_t BENCH_POINTS = 200'000;

void bench::ExpVCL(const float *x, float *result) {
    vcl::Vec16f x_vec;
    for (size_t i = 0; i < BENCH_POINTS; i += vcl::Vec16f::size()) {
        x_vec.load(x + i);
        vcl::exp(x_vec).store(result + i);
    }
}

void bench::SinVCL(const float *x, float *result) {
    vcl::Vec16f x_vec;
    for (size_t i = 0; i < BENCH_POINTS; i += vcl::Vec16f::size()) {
        x_vec.load(x + i);
        vcl::sin(x_vec).store(result + i);
    }
}

void bench::ExpScalar(const float *x, float *result) {
    for (size_t i = 0; i < BENCH_POINTS; ++i) {
        result[i] = expf(x[i]);
    }
}

void bench::SinScalar(const float *x, float *result) {
    for (size_t i = 0; i < BENCH_POINTS; ++i) {
        result[i] = sinf(x[i]);
    }
}
