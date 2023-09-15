#include <iostream>
#include <new>  

#include "lucaskanade.cuh"
#include "kernels.cuh"

LucasKanade::LucasKanade(int spac_filt_size_, int temp_filt_size_, int window_size_, int nx_, int ny_)
{
	nx = nx_;
	ny = ny_;
	spac_filt_size = spac_filt_size_;
	temp_filt_size = temp_filt_size_;
	window_size    = window_size_;
	
	filt_x = new float [spac_filt_size];
	filt_y = new float [spac_filt_size];
	filt_t = new float [temp_filt_size];

	Ix     = new float [nx*ny];
	Iy     = new float [nx*ny];
	It     = new float [nx*ny];

	getfilters(filt_x, filt_y, filt_t, spac_filt_size, temp_filt_size);
	
	cudaMalloc((void**)&d_filt_x, spac_filt_size*sizeof(float));
	cudaMalloc((void**)&d_filt_y, spac_filt_size*sizeof(float));
	cudaMalloc((void**)&d_filt_t, temp_filt_size*sizeof(float));
	cudaMalloc((void**)&d_Ix, nx*ny*sizeof(float));
	cudaMalloc((void**)&d_Iy, nx*ny*sizeof(float));
	cudaMalloc((void**)&d_It, nx*ny*sizeof(float));
	cudaMalloc((void**)&d_wind_frames, temp_filt_size*nx*ny*sizeof(unsigned char));
	cudaMalloc((void**)&d_Vx, nx*ny*sizeof(float));
	cudaMalloc((void**)&d_Vy, nx*ny*sizeof(float));

	cudaMemcpy(d_filt_x, filt_x, spac_filt_size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_filt_y, filt_y, spac_filt_size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_filt_t, filt_t, temp_filt_size*sizeof(float), cudaMemcpyHostToDevice);
}


LucasKanade::~LucasKanade() {
	delete[] filt_x;
	delete[] filt_y;
	delete[] filt_t;

	delete[] Ix;
	delete[] Iy;
	delete[] It;

	cudaFree(d_filt_x);
	cudaFree(d_filt_y);
	cudaFree(d_filt_t);
	cudaFree(d_Ix);
	cudaFree(d_Iy);
	cudaFree(d_wind_frames);
	cudaFree(d_Vx);
	cudaFree(d_Vy);
}


/********************************************************************/
/* Return the Velocity Vx,Vy from an input images in window_frames  */
/********************************************************************/
void LucasKanade::lucas_kanade(float *Vx, float *Vy, int iframe, unsigned char *wind_frames)
{
	unsigned char* frame = wind_frames + (((iframe+1)%temp_filt_size)*nx*ny);
	unsigned char* d_frame = d_wind_frames + (((iframe+1)%temp_filt_size)*nx*ny);

	// It
	temp_convolution_GPU_wrapper(iframe, d_It, d_wind_frames, nx, ny, d_filt_t, temp_filt_size);

	// Ix = imfilter(frame, filter_x,'left-top');
	spac_convolution2D_x_GPU_wrapper(d_Ix, d_frame, nx, ny, d_filt_x, spac_filt_size);

	// Iy = imfilter(frame, filter_y,'left-top');
	spac_convolution2D_y_GPU_wrapper(d_Iy, d_frame, nx, ny, d_filt_y, spac_filt_size);

	luca_kanade_1step_GPU_wrapper(Vx, Vy, d_Vx, d_Vy, d_Ix, d_Iy, d_It,
		spac_filt_size, temp_filt_size, window_size, nx, ny);
}


void LucasKanade::copy_frames_circularbuffer_GPU_wrapper(unsigned char *frame, int temp_size, int fr, int frame_size){
	copy_frames_circularbuffer_GPU(frame, d_wind_frames, temp_size, fr, frame_size);
}


void LucasKanade::getfilters(float *filt_x, float *filt_y, float *filt_t, int spac_filt_size, int temp_filt_size)
{
	float filt[9];
	int i;
	
	if (spac_filt_size==1){
		filt[0] = 1;
	} else if (spac_filt_size==3){
		filt[0] = -1.0/2.0;
		filt[1] =    0;
		filt[2] =  1.0/2.0;
	} else if (spac_filt_size==5){
		filt[0] =  1.0/12.0;
		filt[1] = -2.0/3.0;
		filt[2] =    0;
		filt[3] =  2.0/3.0;
		filt[4] = -1.0/12.0;
	} else if (spac_filt_size==7){
		filt[0] = -1.0/60.0;
		filt[1] =  3.0/20.0;
		filt[2] = -3.0/4.0;
		filt[3] =    0;
		filt[4] =  3.0/4.0;	
		filt[5] = -3.0/20.0;
		filt[6] =  1.0/60.0;
	} else if (spac_filt_size==9){
		filt[0] =  1.0/280.0;
		filt[1] = -4.0/105.0;
		filt[2] =  1.0/5.0;
		filt[3] = -4.0/5.0;
		filt[4] =    0;	
		filt[5] =  4.0/5.0;
		filt[6] = -1.0/5.0;
		filt[7] =  4.0/105.0;
		filt[8] = -1.0/280.0;
	} else 	{
		printf("Not developted!!!\n");
	}

	// Copy filt_x and filt_y
	for (i=0;i<spac_filt_size;i++)
		filt_x[i] = filt_y[i] = filt[i];

	if (temp_filt_size==1){
		filt[0] = 1;
	} else if (temp_filt_size==2){
		filt[0] = -1.0;
		filt[1] =  1.0;
	} else if (temp_filt_size==3){
		filt[0] = -1.0/2.0;
		filt[1] =    0;
		filt[2] =  1.0/2.0;
	} else if (temp_filt_size==5){
		filt[0] =  1.0/12.0;
		filt[1] = -2.0/3.0;
		filt[2] =    0;
		filt[3] =  2.0/3.0;
		filt[4] = -1.0/12.0;
	} else if (temp_filt_size==7){
		filt[0] = -1.0/60.0;
		filt[1] =  3.0/20.0;
		filt[2] = -3.0/4.0;
		filt[3] =    0;
		filt[4] =  3.0/4.0;	
		filt[5] = -3.0/20.0;
		filt[6] =  1.0/60.0;
	} else if (temp_filt_size==9){
		filt[0] =  1.0/280.0;
		filt[1] = -4.0/105.0;
		filt[2] =  1.0/5.0;
		filt[3] = -4.0/5.0;
		filt[4] =    0;	
		filt[5] =  4.0/5.0;
		filt[6] = -1.0/5.0;
		filt[7] =  4.0/105.0;
		filt[8] = -1.0/280.0;
	} else 	{
		printf("Not developted!!!\n");
	}
	// Copy filt_t
	for (i=0;i<temp_filt_size;i++)
		filt_t[i] = filt[i];
}
