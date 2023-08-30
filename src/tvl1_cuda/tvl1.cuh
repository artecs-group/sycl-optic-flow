#ifndef DUAL_TVL1_OPTIC_FLOW_H
#define DUAL_TVL1_OPTIC_FLOW_H

#include <cstdint>
#include <cublas_v2.h>
#include <cuda_fp16.h>

constexpr size_t MAX_ITERATIONS{10};
constexpr float PRESMOOTHING_SIGMA{0.8f};
constexpr float ZOOM_SIGMA_ZERO{0.6f};
constexpr size_t DEFAULT_GAUSSIAN_WINDOW_SIZE{5};

class TV_L1 { 
public:
    TV_L1(int width, int height, float tau=0.25, float lambda=0.15, 
        float theta=0.3, int nscales=100, float zfactor=0.5, int warps=5,
        float epsilon=0.01);
    ~TV_L1();
    const __half2* getU() { return _hostU; };
    void runDualTVL1Multiscale(const __half2* I);

private:
    void dualTVL1(const __half2* I0, const __half2* I1, __half2* u1, __half2* u2, int nx, int ny);
    void imageNormalization(const __half2 *I0, const __half2 *I1, __half2* I0n, __half2* I1n, int size);
    void zoomOut(const __half2 *I, __half2 *Iout, __half2* B, const int* nx, const int* ny, int* nxx, int* nyy, const __half2 factor, __half2* Is, __half2* gaussBuffer, cublasHandle_t* handle);
    void zoomIn(const __half2 *I, __half2 *Iout, const int* nx, const int* ny, const int* nxx, const int* nyy);
    void divergence(const __half2 *v1, const __half2 *v2, __half2 *div, const int nx, const int ny);
    void forwardGradient(const __half2 *f, __half2 *fx, __half2 *fy, const int nx, const int ny);
    void centeredGradient(const __half2* input, __half2 *dx, __half2 *dy, const int nx, const int ny);
    void gaussian(__half2 *I, __half2* B, const int* xdim, const int* ydim, __half2 sigma, __half2* buffer, cublasHandle_t* handle);

    cublasHandle_t _handle;

    __half2* _hostU;

	__half2 *_I0s, *_I1s, *_u1s, *_u2s, *_imBuffer;
	int *_nx, *_ny, *_nxy, *_hNx, *_hNy;

	__half2 *_I1x, *_I1y, *_I1w, *_I1wx, *_I1wy, *_rho_c, *_v1, *_v2, *_p11, *_p12, 
        *_p21, *_p22, *_grad, *_div_p1, *_div_p2, *_g1, *_g2, *_B, *_error;

    int _width;     // image width
    int _height;     // image height
    __half2 _tau;     // time step
    __half2 _lambda;  // weight parameter for the data term
    __half2 _theta;   // weight parameter for (u - v)²
    int _nscales; // number of scales
    __half2 _zfactor; // factor for building the image piramid
    int _warps;   // number of warpings per scale
    __half2 _epsilon; // tolerance for numerical convergence
};

#endif