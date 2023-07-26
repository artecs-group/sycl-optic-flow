#ifndef _LUCASKANADE_H_
#define _LUCASKANADE_H_

#include "kernels.cuh"

class LucasKanade {
	int nx, ny;
	int spac_filt_size;
	int temp_filt_size;
	int window_size;
	
	float *filt_x;
	float *filt_y;
	float *filt_t;
	
	float *Ix;
	float *Iy;
	float *It;
//	*Vx     = new float [nframes*nx*ny]; //get_memory1D_float( nframes*nx*ny );
//	*Vy     = new float [nframes*nx*ny]; //get_memory1D_float( nframes*nx*ny );

	/* For GPU memory */
	float *d_filt_x;
	float *d_filt_y;
	float *d_filt_t;
	float *d_Ix;
	float *d_Iy;
	float *d_It;

	u_char *d_wind_frames;
	float *d_Vx;
	float *d_Vy;
	

public:
	LucasKanade(int spac_filt_size_, int temp_filt_size_, int window_size_, int nx_, int ny_, int gpu_enable);

	void lucas_kanade(float *Vx, float *Vy, int iframe, unsigned char *wind_frames, int gpu);

	void copy_frames_circularbuffer_GPU_wrapper(u_char *frame, int temp_size, int fr, int frame_size);

	~LucasKanade()
	{
		delete[] filt_x;
		delete[] filt_y;
		delete[] filt_t;

		delete[] Ix;
		delete[] Iy;
		delete[] It;

		delete_GPU(d_filt_x, d_filt_y, d_filt_t, d_Ix, d_Iy, d_It, d_wind_frames, d_Vx, d_Vy);
	}
};




#endif
