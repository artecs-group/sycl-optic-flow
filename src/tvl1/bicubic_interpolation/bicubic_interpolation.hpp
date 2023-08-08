#ifndef BICUBIC_INTERPOLATION_C
#define BICUBIC_INTERPOLATION_C

static int neumann_bc(int x, int nx, bool *out);
float bicubic_interpolation_at(const float* input, const float uu, const float vv, const int nx, const int ny, bool border_out);
void bicubic_interpolation_warp(const float* input, const float *u, const float *v, float *output, const int nx, const int ny, bool border_out);

#endif