#ifndef KERNELS
#define KERNELS

#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdint>

constexpr float GRAD_IS_ZERO{1E-10};

SYCL_EXTERNAL void bodyGradient(const float *input, float *dx, float *dy,
                                int nx, int ny,
                                const sycl::nd_item<3> &item_ct1);
SYCL_EXTERNAL void edgeRowsGradient(const float *input, float *dx, float *dy,
                                    int nx, int ny,
                                    const sycl::nd_item<3> &item_ct1);
SYCL_EXTERNAL void edgeColumnsGradient(const float *input, float *dx, float *dy,
                                       int nx, int ny,
                                       const sycl::nd_item<3> &item_ct1);
SYCL_EXTERNAL void cornersGradient(const float *input, float *dx, float *dy,
                                   int nx, int ny);
SYCL_EXTERNAL void convolution1D(float *B, int size, float sPi, float den,
                                 const sycl::nd_item<3> &item_ct1);
SYCL_EXTERNAL void lineConvolution(float *I, const float *B, const int *xDim,
                                   const int *yDim, int size, float *buffer,
                                   const sycl::nd_item<3> &item_ct1);
SYCL_EXTERNAL void columnConvolution(float *I, const float *B, const int *xDim,
                                     const int *yDim, int size, float *buffer,
                                     const sycl::nd_item<3> &item_ct1);

SYCL_EXTERNAL void cornersDivergence(const float *v1, const float *v2,
                                     float *div, int nx, int ny);
SYCL_EXTERNAL void edgeColumnsDivergence(const float *v1, const float *v2,
                                         float *div, int nx, int ny,
                                         const sycl::nd_item<3> &item_ct1);
SYCL_EXTERNAL void edgeRowsDivergence(const float *v1, const float *v2,
                                      float *div, int nx, int ny,
                                      const sycl::nd_item<3> &item_ct1);
SYCL_EXTERNAL void bodyDivergence(const float *v1, const float *v2, float *div,
                                  int nx, int ny,
                                  const sycl::nd_item<3> &item_ct1);

SYCL_EXTERNAL void bodyForwardGradient(const float *f, float *fx, float *fy,
                                       size_t nx, size_t ny,
                                       const sycl::nd_item<3> &item_ct1);
SYCL_EXTERNAL void rowsForwardGradient(const float *f, float *fx, float *fy,
                                       size_t nx, size_t ny,
                                       const sycl::nd_item<3> &item_ct1);
SYCL_EXTERNAL void columnsForwardGradient(const float *f, float *fx, float *fy,
                                          size_t nx, size_t ny,
                                          const sycl::nd_item<3> &item_ct1);

SYCL_EXTERNAL void zoomSize(const int *nx, const int *ny, int *nxx, int *nyy,
                            float factor);
SYCL_EXTERNAL void bicubicResample(const float *Is, float *Iout, const int *nxx,
                                   const int *nyy, const int *nx, const int *ny,
                                   float factor,
                                   const sycl::nd_item<3> &item_ct1);
SYCL_EXTERNAL void bicubicResample2(const float *Is, float *Iout,
                                    const int *nxx, const int *nyy,
                                    const int *nx, const int *ny,
                                    const sycl::nd_item<3> &item_ct1);

float bicubicInterpolationAt(const float* input, float uu, float vv, int nx, int ny, bool border_out);
void bicubicInterpolationWarp(const float *input, const float *u,
                                            const float *v, float *output,
                                            int nx, int ny, bool border_out, int blocks, int threads, sycl::queue queue);

void normKernel(const float *__restrict__ I0,
                              const float *__restrict__ I1,
                              float *__restrict__ I0n, float *__restrict__ I1n,
                              float min, float den, int size,
                              int blocks, int threads, sycl::queue queue);
void calculateRhoGrad(const float *I1wx, const float *I1wy,
                                    const float *I1w, const float *u1,
                                    const float *u2, const float *I0,
                                    float *grad, float *rho_c, int size,
                                    int blocks, int threads, sycl::queue queue);
void estimateThreshold(const float *rho_c, const float *I1wx,
                                     const float *u1, const float *I1wy,
                                     const float *u2, const float *grad,
                                     float lT, size_t size, float *v1,
                                     float *v2, int blocks, int threads, sycl::queue queue);
void estimateOpticalFlow(float *u1, float *u2, const float *v1,
                                       const float *v2, const float *div_p1,
                                       const float *div_p2, float theta,
                                       size_t size, float *error,
                                       int blocks, int threads, sycl::queue queue);
void estimateGArgs(const float *div_p1, const float *div_p2,
                                 const float *v1, const float *v2, size_t size,
                                 float taut, float *g1, float *g2,
                                 int blocks, int threads, sycl::queue queue);
void divideByG(const float *g1, const float *g2, size_t size,
                             float *p11, float *p12, float *p21, float *p22,
                             int blocks, int threads, sycl::queue queue);

#endif