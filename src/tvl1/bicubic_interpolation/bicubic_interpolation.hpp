#ifndef BICUBIC_INTERPOLATION_C
#define BICUBIC_INTERPOLATION_C

#define BOUNDARY_CONDITION 0

static int neumann_bc(int x, int nx, bool *out);
static int periodic_bc(int x, int nx, bool *out);
static int symmetric_bc(int x, int nx, bool *out);
static double cubic_interpolation_cell(double v[4], double x);
static double bicubic_interpolation_cell(double p[4][4], double x, double y);

template<typename T>
float bicubic_interpolation_at(const T *input, const float uu, const float vv, const int nx, const int ny, bool border_out);

template<typename T>
void bicubic_interpolation_warp(const T *input, const float *u, const float *v, float *output, const int nx, const int ny, bool border_out);

#endif