#pragma once

#include "cppbench.hpp"

namespace bench {
#define DEF_VCL_BENCHES(func)            \
    CPP_BENCH_FUNC_DECL(func, f32, vcl); \
    CPP_BENCH_FUNC_DECL(func, f64, vcl);

DEF_VCL_BENCHES(exp)
DEF_VCL_BENCHES(exp2)
DEF_VCL_BENCHES(exp_m1)
DEF_VCL_BENCHES(sin)
DEF_VCL_BENCHES(cos)
DEF_VCL_BENCHES(tan)
DEF_VCL_BENCHES(asin)
DEF_VCL_BENCHES(acos)
DEF_VCL_BENCHES(atan)

#undef DEF_VCL_BENCHES
}  // namespace bench
