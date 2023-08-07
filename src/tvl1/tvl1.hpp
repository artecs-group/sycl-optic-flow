#ifndef DUAL_TVL1_OPTIC_FLOW_H
#define DUAL_TVL1_OPTIC_FLOW_H

#include <cstdint>

#define MAX_ITERATIONS 300
#define PRESMOOTHING_SIGMA 0.8
#define GRAD_IS_ZERO 1E-10

class TV_L1 { 
public:
    TV_L1(int width, int height, float tau=0.25, float lambda=0.15, 
        float theta=0.3, int nscales=100, float zfactor=0.5, int warps=5,
        float epsilon=0.01);
    ~TV_L1();
    float* getU() { return _u; };
    void runDualTVL1Multiscale(uint8_t *I0, uint8_t *I1);

private:
    void dualTVL1(const float* I0, const float* I1, float* u1, float* u2, int nx, int ny);
    void image_normalization(const float *I0, const float *I1, float* I0n, float* I1n, int size);
    void convertToFloat(int size, const uint8_t *I0, const uint8_t *I1, float *_I0, float* _I1);

    float* _u;           // x, y component of the optical flow

	float **_I0s, **_I1s, **_u1s, **_u2s;
	int *_nx, *_ny;

	float *_I1x, *_I1y, *_I1w, *_I1wx, *_I1wy, *_rho_c, *_v1, *_v2, *_p11, *_p12, 
        *_p21, *_p22, *_grad, *_div_p1, *_div_p2;

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