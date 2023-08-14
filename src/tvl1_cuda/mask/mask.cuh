#ifndef MASK
#define MASK

#include <cmath>
#include <cstdint>
#include <cublas_v2.h>

constexpr size_t DEFAULT_GAUSSIAN_WINDOW_SIZE{5};

void divergence(const float *v1, const float *v2, float *div, const int nx, const int ny);
void forward_gradient(const float *f, float *fx, float *fy, const int nx, const int ny);
void centered_gradient(const float* input, float *dx, float *dy, const int nx, const int ny);
void gaussian(float *I, float* B, const int* xdim, const int* ydim, const double sigma, cublasHandle_t* handle);

__global__ void bodyGradient(const float* input, float* dx, float* dy, int nx, int ny);
__global__ void edgeRowsGradient(const float* input, float* dx, float* dy, int nx, int ny);
__global__ void edgeColumnsGradient(const float* input, float* dx, float* dy, int nx, int ny);
__global__ void cornersGradient(const float* input, float* dx, float* dy, int nx, int ny);
__global__ void convolution1D(float* B, int size, float sPi, float den);
__global__ void lineConvolution(const float *I, float *output, const float *B, int xdim, int ydim, int size, int bdx);
__global__ void columnConvolution(const float* I, float* output, const float* B, int xdim, int ydim, int size, int bdy);

#endif