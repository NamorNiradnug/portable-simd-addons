#include "vclbench.hpp"

#define VCL_NAMESPACE vcl
#include "vcl/vectormath_exp.h"
#include "vcl/vectormath_trig.h"

static constexpr size_t BENCH_POINTS = 200'000;

using Vec16fFunc = vcl::Vec16f (*)(vcl::Vec16f);

namespace {
template <Vec16fFunc func>
void vcl_bench_f32_impl(const float *x, float *result) {
    vcl::Vec16f x_vec;
    for (size_t i = 0; i < BENCH_POINTS; i += vcl::Vec16f::size()) {
        x_vec.load(x + i);
        func(x_vec).store(result + i);
    }
}

template <float (*func)(float)>
void scalar_bench_f32_impl(const float *x, float *result) {
    for (size_t i = 0; i < BENCH_POINTS; ++i) {
        result[i] = func(x[i]);
    }
}
}  // namespace

namespace bench {

#define IMPL_BENCH_F32_VCL(func) \
    void func##_f32_vcl(const float *x, float *result) { vcl_bench_f32_impl<vcl::func>(x, result); }

IMPL_BENCH_F32_VCL(exp)
IMPL_BENCH_F32_VCL(exp2)
IMPL_BENCH_F32_VCL(expm1)
IMPL_BENCH_F32_VCL(sin)
IMPL_BENCH_F32_VCL(cos)
IMPL_BENCH_F32_VCL(tan)
IMPL_BENCH_F32_VCL(asin)
IMPL_BENCH_F32_VCL(acos)
IMPL_BENCH_F32_VCL(atan)

#undef IMPL_BENCH_F32_VCL

void atan2_f32_vcl(const float *x, const float *y, float *result) {
    vcl::Vec16f x_vec;
    vcl::Vec16f y_vec;
    for (size_t i = 0; i < BENCH_POINTS; i += vcl::Vec16f::size()) {
        x_vec.load(x + i);
        y_vec.load(y + i);
        vcl::atan2(y_vec, x_vec).store(result + i);
    }
}

#define IMPL_BENCH_F32_SCALAR(func) \
    void func##_f32_scalar(const float *x, float *result) { scalar_bench_f32_impl<func##f>(x, result); }

IMPL_BENCH_F32_SCALAR(exp)
IMPL_BENCH_F32_SCALAR(exp2)
IMPL_BENCH_F32_SCALAR(expm1)
IMPL_BENCH_F32_SCALAR(sin)
IMPL_BENCH_F32_SCALAR(cos)
IMPL_BENCH_F32_SCALAR(tan)
IMPL_BENCH_F32_SCALAR(asin)
IMPL_BENCH_F32_SCALAR(acos)
IMPL_BENCH_F32_SCALAR(atan)

#undef IMPL_BENCH_F32_SCALAR

}  // namespace bench
