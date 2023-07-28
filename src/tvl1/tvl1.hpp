#ifndef DUAL_TVL1_OPTIC_FLOW_H
#define DUAL_TVL1_OPTIC_FLOW_H

#include <cstdint>

#define MAX_ITERATIONS 300
#define PRESMOOTHING_SIGMA 0.8
#define GRAD_IS_ZERO 1E-10

template<typename T>
void Dual_TVL1_optic_flow(
    T *I0,           // source image
    T *I1,           // target image
    float *u1,           // x component of the optical flow
    float *u2,           // y component of the optical flow
    const int   nx,      // image width
    const int   ny,      // image height
    const float tau,     // time step
    const float lambda,  // weight parameter for the data term
    const float theta,   // weight parameter for (u - v)²
    const int   warps,   // number of warpings per scale
    const float epsilon, // tolerance for numerical convergence
    const bool  verbose  // enable/disable the verbose mode
);

static void getminmax(
    float *min,     // output min
    float *max,     // output max
    const uint8_t *x, // input array
    int n           // array size
);

void image_normalization(
    const uint8_t *I0,  // input image0
    const uint8_t *I1,  // input image1
    float *I0n,       // normalized output image0
    float *I1n,       // normalized output image1
    int size          // size of the image
);

void Dual_TVL1_optic_flow_multiscale(
    uint8_t *I0,           // source image
    uint8_t *I1,           // target image
    float *u1,           // x component of the optical flow
    float *u2,           // y component of the optical flow
    const int   nxx,     // image width
    const int   nyy,     // image height
    const float tau,     // time step
    const float lambda,  // weight parameter for the data term
    const float theta,   // weight parameter for (u - v)²
    const int   nscales, // number of scales
    const float zfactor, // factor for building the image piramid
    const int   warps,   // number of warpings per scale
    const float epsilon, // tolerance for numerical convergence
    const bool  verbose  // enable/disable the verbose mode
);

void fixFlowVector(int n, int pd, const float* in, float* out);

#endif