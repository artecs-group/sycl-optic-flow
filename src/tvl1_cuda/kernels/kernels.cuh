#ifndef KERNELS
#define KERNELS

#include <cmath>
#include <cstdint>
#include <cublas_v2.h>
#include <cuda_fp16.h>

constexpr float GRAD_IS_ZERO{1E-10};
constexpr int WRAP_SIZE{32};
constexpr int WRAPS_PER_BLOCK{THREADS_PER_BLOCK/WRAP_SIZE};
constexpr unsigned int MASK{0xffffffff};

__global__ void bodyGradient(const __half2* input, __half2* dx, __half2* dy, int nx, int ny);
__global__ void edgeRowsGradient(const __half2* input, __half2* dx, __half2* dy, int nx, int ny);
__global__ void edgeColumnsGradient(const __half2* input, __half2* dx, __half2* dy, int nx, int ny);
__global__ void cornersGradient(const __half2* input, __half2* dx, __half2* dy, int nx, int ny);
__global__ void convolution1D(float* B, int size, float sPi, float den);
__global__ void lineConvolution(__half2 *I, const float* B, const int* xDim, const int* yDim, int size, __half2* buffer);
__global__ void columnConvolution(__half2* I, const float* B, const int* xDim, const int* yDim, int size, __half2* buffer);

__global__ void cornersDivergence(const __half2* v1, const __half2* v2, __half2* div, int nx, int ny);
__global__ void edgeColumnsDivergence(const __half2* v1, const __half2* v2, __half2* div, int nx, int ny);
__global__ void edgeRowsDivergence(const __half2* v1, const __half2* v2, __half2* div, int nx, int ny);
__global__ void bodyDivergence(const __half2* v1, const __half2* v2, __half2* div, int nx, int ny);

__global__ void bodyForwardGradient(const __half2* f, __half2* fx, __half2* fy, size_t nx, size_t ny);
__global__ void rowsForwardGradient(const __half2* f, __half2* fx, __half2* fy, size_t nx, size_t ny);
__global__ void columnsForwardGradient(const __half2* f, __half2* fx, __half2* fy, size_t nx, size_t ny);

__global__ void zoomSize(const int* nx, const int* ny, int* nxx, int* nyy, __half2 factor);
__global__ void bicubicResample(const __half2* Is, __half2 *Iout, const int* nxx, const int* nyy, const int* nx, const int* ny, __half2 factor);
__global__ void bicubicResample2(const __half2* Is, __half2 *Iout, const int* nxx, const int* nyy, const int* nx, const int* ny);

__device__ __half2 bicubicInterpolationAt(const __half2* input, __half2 uu, __half2 vv, int nx, int ny, bool border_out);
__global__ void bicubicInterpolationWarp(const __half2* input, const __half2 *u, const __half2 *v, __half2 *output, int nx, int ny, bool border_out);

__global__ void normKernel(const __half2* __restrict__ I0, const __half2* __restrict__ I1, __half2* __restrict__ I0n, __half2* __restrict__ I1n, __half2 min, __half2 den, int size);
__global__ void calculateRhoGrad(const __half2* I1wx, const __half2* I1wy, const __half2* I1w, const __half2* u1, const __half2* u2, const __half2* I0, __half2* grad, __half2* rho_c, int size);
__global__ void estimateThreshold(const __half2* rho_c, const __half2* I1wx, const __half2* u1, const __half2* I1wy, const __half2* u2, const __half2* grad, __half2 lT, size_t size, __half2* v1, __half2* v2);
__global__ void estimateOpticalFlow(__half2* u1, __half2* u2, const __half2* v1, const __half2* v2, const __half2* div_p1, const __half2* div_p2, __half2 theta, size_t size, float* error);
__global__ void estimateGArgs(const __half2* div_p1, const __half2* div_p2, const __half2* v1, const __half2* v2, size_t size, __half2 taut, __half2* g1, __half2* g2);
__global__ void divideByG(const __half2* g1, const __half2* g2, size_t size, __half2* p11, __half2* p12, __half2* p21, __half2* p22);

__global__ void copyFloat2Half2(const float* __restrict__ in, __half2* out, int size);

__device__ __half2 warpMax(__half2 max);
__device__ __half2 warpMin(__half2 min);
__device__ bool lastBlock(int* counter);
__global__ void half2MaxMin(int N, __half2* __restrict__ inVec, __half2* __restrict__ partialMax, __half2* __restrict__ partialMin, int* __restrict__ lastBlockCounter);

#endif