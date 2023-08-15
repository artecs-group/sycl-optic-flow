#ifndef DUAL_TVL1_OPTIC_FLOW_H
#define DUAL_TVL1_OPTIC_FLOW_H

#include <cstdint>
#include <cublas_v2.h>

constexpr size_t MAX_ITERATIONS{300};
constexpr float PRESMOOTHING_SIGMA{0.8f};
constexpr float ZOOM_SIGMA_ZERO{0.6};
constexpr size_t DEFAULT_GAUSSIAN_WINDOW_SIZE{5};

class TV_L1 { 
public:
    TV_L1(int width, int height, float tau=0.25, float lambda=0.15, 
        float theta=0.3, int nscales=100, float zfactor=0.5, int warps=5,
        float epsilon=0.01);
    ~TV_L1();
    const float* getU() { return _hostU; };
    void runDualTVL1Multiscale(const float *I0, const float *I1);

private:
    void dualTVL1(const float* I0, const float* I1, float* u1, float* u2, int nx, int ny);
    void imageNormalization(const float *I0, const float *I1, float* I0n, float* I1n, int size);
    void zoomOut(const float *I, float *Iout, float* B, const int* nx, const int* ny, int* nxx, int* nyy, const float factor, float* Is, cublasHandle_t* handle);
    void zoomIn(const float *I, float *Iout, const int* nx, const int* ny, const int* nxx, const int* nyy);
    void divergence(const float *v1, const float *v2, float *div, const int nx, const int ny);
    void forwardGradient(const float *f, float *fx, float *fy, const int nx, const int ny);
    void centeredGradient(const float* input, float *dx, float *dy, const int nx, const int ny);
    void gaussian(float *I, float* B, const int* xdim, const int* ydim, const double sigma, cublasHandle_t* handle);

    cublasHandle_t _handle;

    float* _hostU;

	float *_I0s, *_I1s, *_u1s, *_u2s;
	int *_nx, *_ny, *_nxy, *_hNx, *_hNy;

	float *_I1x, *_I1y, *_I1w, *_I1wx, *_I1wy, *_rho_c, *_v1, *_v2, *_p11, *_p12, 
        *_p21, *_p22, *_grad, *_div_p1, *_div_p2, *_g1, *_g2, *_B, *_error;

    int _width;     // image width
    int _height;     // image height
    float _tau;     // time step
    float _lambda;  // weight parameter for the data term
    float _theta;   // weight parameter for (u - v)²
    int _nscales; // number of scales
    float _zfactor; // factor for building the image piramid
    int _warps;   // number of warpings per scale
    float _epsilon; // tolerance for numerical convergence
};

#endif