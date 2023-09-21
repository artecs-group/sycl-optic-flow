#include <sycl/sycl.hpp>
#include <iostream>

#include "lucaskanade.hpp"

LucasKanade::LucasKanade(sycl::queue queue, int spac_filt_size_, int temp_filt_size_, int window_size_, int nx_, int ny_)
{
	_queue = queue;
    nx = nx_;
	ny = ny_;
	spac_filt_size = spac_filt_size_;
	temp_filt_size = temp_filt_size_;
	window_size    = window_size_;
	
	filt_x = new float [spac_filt_size];
	filt_y = new float [spac_filt_size];
	filt_t = new float [temp_filt_size];

	getfilters(filt_x, filt_y, filt_t, spac_filt_size, temp_filt_size);

	d_filt_x = sycl::malloc_device<float>(spac_filt_size, _queue);
	d_filt_y = sycl::malloc_device<float>(spac_filt_size, _queue);
	d_filt_t = sycl::malloc_device<float>(temp_filt_size, _queue);
	d_Ix = sycl::malloc_device<float>(nx * ny, _queue);
	d_Iy = sycl::malloc_device<float>(nx * ny, _queue);
	d_It = sycl::malloc_device<float>(nx * ny, _queue);
	d_wind_frames = sycl::malloc_device<unsigned char>(temp_filt_size * nx * ny, _queue);
	d_Vx = sycl::malloc_device<float>(nx * ny, _queue);
	d_Vy = sycl::malloc_device<float>(nx * ny, _queue);

	_queue.memcpy(d_filt_x, filt_x, spac_filt_size * sizeof(float));
	_queue.memcpy(d_filt_y, filt_y, spac_filt_size * sizeof(float));
	_queue.memcpy(d_filt_t, filt_t, temp_filt_size * sizeof(float));
	_queue.wait();
}

LucasKanade::~LucasKanade() {
    delete[] filt_x;
	delete[] filt_y;
	delete[] filt_t;

	sycl::free(d_filt_x, _queue);
	sycl::free(d_filt_y, _queue);
	sycl::free(d_filt_t, _queue);
	sycl::free(d_Ix, _queue);
	sycl::free(d_Iy, _queue);
	sycl::free(d_wind_frames, _queue);
	sycl::free(d_Vx, _queue);
	sycl::free(d_Vy, _queue);
}


/********************************************************************/
/* Return the Velocity Vx,Vy from an input images in window_frames  */
/********************************************************************/
void LucasKanade::lucas_kanade(float *Vx, float *Vy, int iframe)
{
	unsigned char* d_frame = d_wind_frames + (((iframe+1)%temp_filt_size)*nx*ny);

	// It
	temp_convolution_GPU_wrapper(_queue, iframe, d_It, d_wind_frames, nx, ny, d_filt_t, temp_filt_size);

	// Ix = imfilter(frame, filter_x,'left-top');
	spac_convolution2D_x_GPU_wrapper(_queue, d_Ix, d_frame, nx, ny, d_filt_x, spac_filt_size);

	// Iy = imfilter(frame, filter_y,'left-top');
	spac_convolution2D_y_GPU_wrapper(_queue, d_Iy, d_frame, nx, ny, d_filt_y, spac_filt_size);
	std::cout << "kk2" << std::endl;

	luca_kanade_1step_GPU_wrapper(_queue, Vx, Vy, d_Vx, d_Vy, d_Ix, d_Iy, d_It,
		spac_filt_size, temp_filt_size, window_size, nx, ny);
	std::cout << "kk3" << std::endl;
}


void LucasKanade::copy_frames_circularbuffer_GPU_wrapper(unsigned char *frame, int temp_size, int fr, int frame_size){
	copy_frames_circularbuffer_GPU(_queue, frame, d_wind_frames, temp_size, fr, frame_size);
}


void LucasKanade::getfilters(float *filt_x, float *filt_y, float *filt_t, int spac_filt_size, int temp_filt_size)
{
	float filt[9];
	int i;
	
	if (spac_filt_size==1){
		filt[0] = 1.0f;
	} else if (spac_filt_size==3){
		filt[0] = -1.0f/2.0f;
		filt[1] =    0.0f;
		filt[2] =  1.0f/2.0f;
	} else if (spac_filt_size==5){
		filt[0] =  1.0f/12.0f;
		filt[1] = -2.0f/3.0f;
		filt[2] =    0.0f;
		filt[3] =  2.0f/3.0f;
		filt[4] = -1.0f/12.0f;
	} else if (spac_filt_size==7){
		filt[0] = -1.0f/60.0f;
		filt[1] =  3.0f/20.0f;
		filt[2] = -3.0f/4.0f;
		filt[3] =    0.0f;
		filt[4] =  3.0f/4.0f;	
		filt[5] = -3.0f/20.0f;
		filt[6] =  1.0f/60.0f;
	} else if (spac_filt_size==9){
		filt[0] =  1.0f/280.0f;
		filt[1] = -4.0f/105.0f;
		filt[2] =  1.0f/5.0f;
		filt[3] = -4.0f/5.0f;
		filt[4] =    0.0f;	
		filt[5] =  4.0f/5.0f;
		filt[6] = -1.0f/5.0f;
		filt[7] =  4.0f/105.0f;
		filt[8] = -1.0f/280.0f;
	} else 	{
		printf("Not developted!!!\n");
	}

	// Copy filt_x and filt_y
	for (i=0;i<spac_filt_size;i++)
		filt_x[i] = filt_y[i] = filt[i];

	if (temp_filt_size==1){
		filt[0] = 1.0f;
	} else if (temp_filt_size==2){
		filt[0] = -1.0f;
		filt[1] =  1.0f;
	} else if (temp_filt_size==3){
		filt[0] = -1.0f/2.0f;
		filt[1] =    0.0f;
		filt[2] =  1.0f/2.0f;
	} else if (temp_filt_size==5){
		filt[0] =  1.0f/12.0f;
		filt[1] = -2.0f/3.0f;
		filt[2] =    0.0f;
		filt[3] =  2.0f/3.0f;
		filt[4] = -1.0f/12.0f;
	} else if (temp_filt_size==7){
		filt[0] = -1.0f/60.0f;
		filt[1] =  3.0f/20.0f;
		filt[2] = -3.0f/4.0f;
		filt[3] =    0.0f;
		filt[4] =  3.0f/4.0f;	
		filt[5] = -3.0f/20.0f;
		filt[6] =  1.0f/60.0f;
	} else if (temp_filt_size==9){
		filt[0] =  1.0f/280.0f;
		filt[1] = -4.0f/105.0f;
		filt[2] =  1.0f/5.0f;
		filt[3] = -4.0f/5.0f;
		filt[4] =    0.0f;	
		filt[5] =  4.0f/5.0f;
		filt[6] = -1.0f/5.0f;
		filt[7] =  4.0f/105.0f;
		filt[8] = -1.0f/280.0f;
	} else 	{
		printf("Not developted!!!\n");
	}
	// Copy filt_t
	for (i=0;i<temp_filt_size;i++)
		filt_t[i] = filt[i];
}
