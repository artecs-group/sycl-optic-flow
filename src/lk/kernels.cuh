#ifndef KERNELS_CUH
#define KERNELS_CUH

void init_GPU(float **d_filt_x_, float **d_filt_y_, float **d_filt_t_, float **d_Ix_, float **d_Iy_, float **d_It_, 
	u_char **d_wind_frames_, float **d_Vx, float **d_Vy,
	float *filt_x, float *filt_y, float *filt_t, 
	int spac_filt_size, int temp_filt_size, int nx, int ny);

void delete_GPU(float *d_filt_x, float *d_filt_y, float *d_filt_t, float *d_Ix, float *d_Iy, float *d_It, u_char *d_wind_frames,
	float *d_Vx, float *d_Vy);

void copy_frames_circularbuffer_GPU(u_char *raw_frame, u_char *d_wind_frames, int temp_conv_size, int fr, int size);

void temp_convolution_GPU_wrapper(int iframe, float *It, unsigned char *frame_in, int nx, int ny, float *filter, int filter_size);
void spac_convolution2D_x_GPU_wrapper(float *Ix, unsigned char *frame, int nx, int ny, float *filter, int filter_size);
void spac_convolution2D_y_GPU_wrapper(float *Iy, unsigned char *frame, int nx, int ny, float *filter, int filter_size);
void luca_kanade_1step_GPU_wrapper(float *Vx, float *Vy, 
	float *d_Vx, float *d_Vy, float *Ix, float *Iy, float *It,
	int spac_filt_size, int temp_filt_size, int window_size, int nx, int ny);
#endif
