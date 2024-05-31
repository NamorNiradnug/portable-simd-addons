#pragma once
#include <cstddef>

using f32 = float;
using f64 = double;

static constexpr size_t BENCH_POINTS = 200'000;

template <class X>
using Func = X (*)(X);

#define CPP_BENCH_FUNC_NAME(func, ftype, suffix) func##_##ftype##_##suffix

#define CPP_BENCH_FUNC_DECL(func, ftype, suffix) \
    void CPP_BENCH_FUNC_NAME(func, ftype, suffix)(const ftype *__restrict__ x, ftype *__restrict__ result)
