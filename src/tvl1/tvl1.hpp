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
        float epsilon=0.01, bool verbose=false);
    ~TV_L1();
    float* getU() { return _u; };
    void runDualTVL1Multiscale(uint8_t *I0, uint8_t *I1);

    template<typename T>
    void dualTVL1(const T* I0, const T* I1, float* u1, float* u2, int nx, int ny);
private:
    float* _u;           // x component of the optical flow
    float* _v;           // y component of the optical flow

	float **_I0s, **_I1s, **_u1s, **_u2s;
	int *_nx, *_ny;

	float *_I1x, *_I1y, *_I1w, *_I1wx, *_I1wy, *_rho_c, *_v1, *_v2, *_p11, *_p12, 
        *_p21, *_p22, *_div, *_grad, *_div_p1, *_div_p2, *_u1x, *_u1y, *_u2x, *_u2y;

    int _width;     // image width
    int _height;     // image height
    float _tau;     // time step
    float _lambda;  // weight parameter for the data term
    float _theta;   // weight parameter for (u - v)²
    int _nscales; // number of scales
    float _zfactor; // factor for building the image piramid
    int _warps;   // number of warpings per scale
    float _epsilon; // tolerance for numerical convergence
    bool _verbose;  // enable/disable the verbose mode
};

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

#endif