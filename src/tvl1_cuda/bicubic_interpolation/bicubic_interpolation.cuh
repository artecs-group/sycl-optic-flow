#ifndef BICUBIC_INTERPOLATION_C
#define BICUBIC_INTERPOLATION_C

__device__ inline int neumann_bc(int x, int nx, bool *out);
__device__ inline float cubic_interpolation_cell(const float v[4], float x);
__device__ inline float bicubic_interpolation_cell(const float p[4][4], float* v, float x, float y);
__device__ float bicubic_interpolation_at(const float* input, float uu, float vv, int nx, int ny, bool border_out);
void bicubic_interpolation_warp(const float* input, const float *u, const float *v, float *output, int nx, int ny, bool border_out);

#endif