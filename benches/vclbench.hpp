#pragma once

namespace bench {
#define DEF_BENCHES_F32(func)                          \
    void func##_f32_vcl(const float *x, float *result); \
    void func##_f32_scalar(const float *x, float *result);

DEF_BENCHES_F32(exp)
DEF_BENCHES_F32(exp2)
DEF_BENCHES_F32(expm1)
DEF_BENCHES_F32(sin)
DEF_BENCHES_F32(cos)
DEF_BENCHES_F32(tan)
DEF_BENCHES_F32(asin)
DEF_BENCHES_F32(acos)
DEF_BENCHES_F32(atan)

#undef DEF_BENCHES_F32
}  // namespace bench
