#pragma once

#include "cppbench.hpp"

namespace bench {
#define DEF_VECLIB_BENCHES(func)         \
    CPP_BENCH_FUNC_DECL(func, f32, libmvec); \
    CPP_BENCH_FUNC_DECL(func, f64, libmvec);

DEF_VECLIB_BENCHES(exp)
DEF_VECLIB_BENCHES(exp2)
DEF_VECLIB_BENCHES(exp_m1)
DEF_VECLIB_BENCHES(sin)
DEF_VECLIB_BENCHES(cos)
DEF_VECLIB_BENCHES(tan)
DEF_VECLIB_BENCHES(asin)
DEF_VECLIB_BENCHES(acos)
DEF_VECLIB_BENCHES(atan)

#undef DEF_VECLIB_BENCHES
}  // namespace bench
