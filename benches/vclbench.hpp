#pragma once

namespace bench {
void exp_f32_vcl(const float *x, float *result);
void sin_f32_vcl(const float *x, float *result);
void asin_f32_vcl(const float *x, float *result);
void atan_f32_vcl(const float *x, float *result);
void atan2_f32_vcl(const float *x, const float *y, float *result);

void exp_f32_scalar(const float *x, float *result);
void sin_f32_scalar(const float *x, float *result);
}  // namespace bench
