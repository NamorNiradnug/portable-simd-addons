#include "vclbench.hpp"

#define VCL_NAMESPACE vcl
#include "vcl/vectormath_exp.h"
#include "vcl/vectormath_trig.h"

namespace {
template <class Scalar, class Vec, Func<Vec> func>
void vcl_bench_impl(const Scalar *__restrict__ x, Scalar *__restrict__ result) {
    Vec x_vec;
    for (size_t i = 0; i < BENCH_POINTS; i += Vec::size()) {
        x_vec.load(x + i);
        func(x_vec).store(result + i);
    }
}
}  // namespace

namespace bench {
#define IMPL_VCL_BECHES(func)                                                                       \
    CPP_BENCH_FUNC_DECL(func, f32, vcl) { vcl_bench_impl<f32, vcl::Vec16f, vcl::func>(x, result); } \
    CPP_BENCH_FUNC_DECL(func, f64, vcl) { vcl_bench_impl<f64, vcl::Vec8d, vcl::func>(x, result); }

IMPL_VCL_BECHES(exp)
IMPL_VCL_BECHES(exp2)
CPP_BENCH_FUNC_DECL(exp_m1, f32, vcl) { vcl_bench_impl<f32, vcl::Vec16f, vcl::expm1>(x, result); }
CPP_BENCH_FUNC_DECL(exp_m1, f64, vcl) { vcl_bench_impl<f64, vcl::Vec8d, vcl::expm1>(x, result); }

IMPL_VCL_BECHES(sin)
IMPL_VCL_BECHES(cos)
IMPL_VCL_BECHES(tan)
IMPL_VCL_BECHES(asin)
IMPL_VCL_BECHES(acos)
IMPL_VCL_BECHES(atan)

void atan2_f32_vcl(const float *x, const float *y, float *result) {
    vcl::Vec16f x_vec;
    vcl::Vec16f y_vec;
    for (size_t i = 0; i < BENCH_POINTS; i += vcl::Vec16f::size()) {
        x_vec.load(x + i);
        y_vec.load(y + i);
        vcl::atan2(y_vec, x_vec).store(result + i);
    }
}

#undef IMPL_VCL_BECHES
}  // namespace bench
