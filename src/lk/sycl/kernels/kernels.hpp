#ifndef KERNELS_CUH
#define KERNELS_CUH

#ifdef ACPP
    #include <CL/sycl.hpp>
    using namespace cl;
#else
    #include <sycl/sycl.hpp>
#endif

void copy_frames_circularbuffer_GPU(sycl::queue queue, unsigned char *raw_frame, unsigned char *d_wind_frames, int temp_conv_size, int fr, int size);

void temp_convolution_GPU_wrapper(sycl::queue queue, int iframe, float *It, unsigned char *frame_in, int nx, int ny, float *filter, int filter_size);
void spac_convolution2D_x_GPU_wrapper(sycl::queue queue, float *Ix, unsigned char *frame, int nx, int ny, float *filter, int filter_size);
void spac_convolution2D_y_GPU_wrapper(sycl::queue queue, float *Iy, unsigned char *frame, int nx, int ny, float *filter, int filter_size);
void luca_kanade_1step_GPU_wrapper(sycl::queue queue, float *Vx, float *Vy, 
	float *d_Vx, float *d_Vy, float *Ix, float *Iy, float *It,
	int spac_filt_size, int temp_filt_size, int window_size, int nx, int ny);
#endif
