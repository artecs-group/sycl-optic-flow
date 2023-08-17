#ifndef KERNELS
#define KERNELS

#include <cmath>
#include <cstdint>
#include <cublas_v2.h>

constexpr float GRAD_IS_ZERO{1E-10};

__global__ void bodyGradient(const float* input, float* dx, float* dy, int nx, int ny);
__global__ void edgeRowsGradient(const float* input, float* dx, float* dy, int nx, int ny);
__global__ void edgeColumnsGradient(const float* input, float* dx, float* dy, int nx, int ny);
__global__ void cornersGradient(const float* input, float* dx, float* dy, int nx, int ny);
__global__ void convolution1D(float* B, int size, float sPi, float den);
__global__ void lineConvolution(float *I, const float *B, const int* xDim, const int* yDim, int size, float* buffer);
__global__ void columnConvolution(float* I, const float* B, const int* xDim, const int* yDim, int size, float* buffer);

__global__ void cornersDivergence(const float* v1, const float* v2, float* div, int nx, int ny);
__global__ void edgeColumnsDivergence(const float* v1, const float* v2, float* div, int nx, int ny);
__global__ void edgeRowsDivergence(const float* v1, const float* v2, float* div, int nx, int ny);
__global__ void bodyDivergence(const float* v1, const float* v2, float* div, int nx, int ny);

__global__ void bodyForwardGradient(const float* f, float* fx, float* fy, size_t nx, size_t ny);
__global__ void rowsForwardGradient(const float* f, float* fx, float* fy, size_t nx, size_t ny);
__global__ void columnsForwardGradient(const float* f, float* fx, float* fy, size_t nx, size_t ny);

__global__ void zoomSize(const int* nx, const int* ny, int* nxx, int* nyy, float factor);
__global__ void bicubicResample(const float* Is, float *Iout, const int* nxx, const int* nyy, const int* nx, const int* ny, float factor);
__global__ void bicubicResample2(const float* Is, float *Iout, const int* nxx, const int* nyy, const int* nx, const int* ny);

__device__ float bicubicInterpolationAt(const float* input, float uu, float vv, int nx, int ny, bool border_out);
__global__ void bicubicInterpolationWarp(const float* input, const float *u, const float *v, float *output, int nx, int ny, bool border_out);

__global__ void normKernel(const float* __restrict__ I0, const float* __restrict__ I1, float* __restrict__ I0n, float* __restrict__ I1n, int min, int den, int size);
__global__ void calculateRhoGrad(const float* I1wx, const float* I1wy, const float* I1w, const float* u1, const float* u2, const float* I0, float* grad, float* rho_c, int size);
__global__ void estimateThreshold(const float* rho_c, const float* I1wx, const float* u1, const float* I1wy, const float* u2, const float* grad, float lT, size_t size, float* v1, float* v2);
__global__ void estimateOpticalFlow(float* u1, float* u2, const float* v1, const float* v2, const float* div_p1, const float* div_p2, float theta, size_t size, float* error);
__global__ void estimateGArgs(const float* div_p1, const float* div_p2, const float* v1, const float* v2, size_t size, float taut, float* g1, float* g2);
__global__ void divideByG(const float* g1, const float* g2, size_t size, float* p11, float* p12, float* p21, float* p22);

#endif