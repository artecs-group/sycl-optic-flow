#ifndef MASK
#define MASK

#include <cmath>
#include <cstdint>
#include <cublas_v2.h>

constexpr size_t DEFAULT_GAUSSIAN_WINDOW_SIZE{5};

void divergence(const float *v1, const float *v2, float *div, const int nx, const int ny);
void forward_gradient(const float *f, float *fx, float *fy, const int nx, const int ny);
void centered_gradient(const float* input, float *dx, float *dy, const int nx, const int ny);
void gaussian(float *I, const int xdim, const int ydim, const double sigma, float* buffer, cublasHandle_t* handle);

#endif