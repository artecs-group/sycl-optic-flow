#ifndef ZOOM_C
#define ZOOM_C

#include <cublas_v2.h>

constexpr float ZOOM_SIGMA_ZERO{0.6};

void zoom_out(const float *I, float *Iout, float* B, const int* nx, const int* ny, int* nxx, int* nyy, const float factor, float* Is, cublasHandle_t* handle);
void zoom_in(const float *I, float *Iout, const int* nx, const int* ny, const int* nxx, const int* nyy);
__global__ void zoom_size(const int* nx, const int* ny, int* nxx, int* nyy, float factor);
__global__ void bicubicResample(const float* Is, float *Iout, const int* nxx, const int* nyy, const int* nx, const int* ny, float factor);
__global__ void bicubicResample2(const float* Is, float *Iout, const int* nxx, const int* nyy, const int* nx, const int* ny);

#endif