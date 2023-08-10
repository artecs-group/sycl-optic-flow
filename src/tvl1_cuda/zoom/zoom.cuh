#ifndef ZOOM_C
#define ZOOM_C

#include <cublas_v2.h>

constexpr float ZOOM_SIGMA_ZERO{0.6};

void zoom_out(const float *I, float *Iout, const int nx, const int ny, const float factor, float* Is, float* gaussBuffer, cublasHandle_t* handle);
void zoom_in(const float *I, float *Iout, int nx, int ny, int nxx, int nyy);
void zoom_size(int nx, int ny, int *nxx, int *nyy, float factor);

#endif