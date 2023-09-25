#ifndef _LUCASKANADE_H_
#define _LUCASKANADE_H_

#include <sycl/sycl.hpp>
#include "kernels/kernels.hpp"

class LucasKanade {
public:
	LucasKanade(sycl::queue queue, int spac_filt_size_, int temp_filt_size_, int window_size_, int nx_, int ny_);
	~LucasKanade();
	void lucas_kanade(float *Vx, float *Vy, int iframe);
	void copy_frames_circularbuffer_GPU_wrapper(unsigned char *frame, int temp_size, int fr, int frame_size);
private:
	void getfilters(float *filt_x, float *filt_y, float *filt_t, int spac_filt_size, int temp_filt_size);
	
	int nx, ny;
	int spac_filt_size;
	int temp_filt_size;
	int window_size;
	
	float *filt_x;
	float *filt_y;
	float *filt_t;

	/* For GPU memory */
	float *d_filt_x;
	float *d_filt_y;
	float *d_filt_t;
	float *d_Ix;
	float *d_Iy;
	float *d_It;

	unsigned char *d_wind_frames;
	float *d_Vx;
	float *d_Vy;

	sycl::queue _queue;
};

#endif
