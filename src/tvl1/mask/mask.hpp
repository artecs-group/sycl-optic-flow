#ifndef MASK
#define MASK

#define BOUNDARY_CONDITION_DIRICHLET 0
#define BOUNDARY_CONDITION_REFLECTING 1
#define BOUNDARY_CONDITION_PERIODIC 2

#define DEFAULT_GAUSSIAN_WINDOW_SIZE 5
#define DEFAULT_BOUNDARY_CONDITION BOUNDARY_CONDITION_REFLECTING

void divergence(const float *v1, const float *v2, float *div, const int nx, const int ny);
void forward_gradient(const float *f, float *fx, float *fy, const int nx, const int ny);

template<typename T>
void centered_gradient(const T *input, float *dx, float *dy, const int nx, const int ny);
void gaussian(float *I, const int xdim, const int ydim, const double sigma);

#endif