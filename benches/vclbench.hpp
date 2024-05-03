#pragma once

namespace bench {
void ExpVCL(const float *x, float *result);
void SinVCL(const float *x, float *result);

void ExpScalar(const float *x, float *result);
void SinScalar(const float *x, float *result);
}  // namespace bench
