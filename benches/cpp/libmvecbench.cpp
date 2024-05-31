#include "libmvecbench.hpp"

#include <cmath>

namespace {
template <class T, Func<T> func>
void libmvec_bench_impl(const T *__restrict__ x, T *__restrict__ result) {
    for (size_t i = 0; i < BENCH_POINTS; ++i) {
        result[i] = func(x[i]);
    }
}
}  // namespace

namespace bench {
#define IMPL_LIBMVEC_BENCHES(func)                                                          \
    CPP_BENCH_FUNC_DECL(func, f32, libmvec) { libmvec_bench_impl<f32, func##f>(x, result); } \
    CPP_BENCH_FUNC_DECL(func, f64, libmvec) { libmvec_bench_impl<f64, func>(x, result); }

IMPL_LIBMVEC_BENCHES(exp)
IMPL_LIBMVEC_BENCHES(exp2)
CPP_BENCH_FUNC_DECL(exp_m1, f32, libmvec) { libmvec_bench_impl<f32, expm1f>(x, result); }
CPP_BENCH_FUNC_DECL(exp_m1, f64, libmvec) { libmvec_bench_impl<f64, expm1>(x, result); }

IMPL_LIBMVEC_BENCHES(sin)
IMPL_LIBMVEC_BENCHES(cos)
IMPL_LIBMVEC_BENCHES(tan)
IMPL_LIBMVEC_BENCHES(asin)
IMPL_LIBMVEC_BENCHES(acos)
IMPL_LIBMVEC_BENCHES(atan)

#undef IMPL_LIBMVEC_BENCHES
}  // namespace benches
