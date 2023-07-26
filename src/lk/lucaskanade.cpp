#include <iostream>
#include <new>  

#include "lucaskanade.hpp"
#include "kernels.cuh"


void getfilters(float *filt_x, float *filt_y, float *filt_t, int spac_filt_size, int temp_filt_size);


LucasKanade::LucasKanade(int spac_filt_size_, int temp_filt_size_, int window_size_, int nx_, int ny_, int gpu_enable)
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

	if (gpu_enable)
		init_GPU(&d_filt_x, &d_filt_y, &d_filt_t, &d_Ix, &d_Iy, &d_It, 
			&d_wind_frames, &d_Vx, &d_Vy,
			filt_x, filt_y, filt_t, 
			spac_filt_size, temp_filt_size,	nx, ny);
}

void getfilters(float *filt_x, float *filt_y, float *filt_t, int spac_filt_size, int temp_filt_size)
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

	

/************************************/
/* temporal convolution             */
/************************************/
void temp_convolution(int iframe, float *im_conv, unsigned char *frame_in, int nx, int ny, float *filter, int filter_size)
{
	int ii,jj,fr;
	unsigned char *im;

	// Init im_conv
	for(ii = 0; ii < ny; ++ii)
		for(jj=0; jj < nx; ++jj) // number of columns
			im_conv[ii*nx + jj] = 0.0;

	for(fr=0; fr<filter_size; fr++){
		im = frame_in + (((fr+iframe+1)%filter_size)*nx*ny);

		for(ii = 0; ii < ny; ++ii)
		{
			for(jj=0; jj < nx; ++jj) // number of columns
			{
				im_conv[ii*nx + jj] += filter[fr]*im[ii*nx + jj];
			}
		}
	}
}


/***************************/
/* convolution 2D centered */
/***************************/
void spac_convolution2D_x(float *im_conv, unsigned char *frame_in, int nx, int ny, float *filter, int filter_size)
{
	int ii, jj, m;
	int bi, bj;
	float pixel;

	float *ptmp;

	int filter_center = (filter_size-1)/2;

	for (ii=0; ii<ny; ii++){

		for (jj=0; jj<filter_center; jj++)
		{
			pixel = 0.0;

			for(m=filter_center; m<=filter_center; ++m)  // full kernel
			{
				if (jj+m>=0)
					pixel += frame_in[ii*nx + jj+m] * filter[m+filter_center];
			}
			im_conv[ii*nx + jj] = pixel;
		}

		// Center
		for (jj=filter_center; jj<nx-filter_center; jj++)	
		{
			pixel = 0.0;

			for(m =-filter_center; m<=filter_center; ++m)  // full kernel
			{
				pixel += frame_in[ii*nx + jj+m] * filter[m+filter_center];
			}
			im_conv[ii*nx + jj] = pixel;
		}

		for (jj=nx-filter_center; jj<nx; jj++)
		{
			pixel = 0.0;

			for(m=filter_center; m<=filter_center; ++m)  // full kernel
			{
				if (jj+m<nx)
					pixel += frame_in[ii*nx + jj+m] * filter[m+filter_center];
			}
			im_conv[ii*nx + jj] = pixel;
		}
	}
}

void spac_convolution2D_y(float *im_conv, unsigned char *frame_in, int nx, int ny, float *filter, int filter_size)
{
	int ii, jj, m;
	int bi, bj;
	float pixel;

	int filter_center = (filter_size-1)/2;

	for (ii=0; ii<filter_center; ii++)
		for (jj=0; jj<nx; jj++)	
		{
			pixel = 0;

			for(m =-filter_center; m<=filter_center; ++m)  // full kernel
			{
				if (ii+m>=0)
					pixel += frame_in[(ii+m)*nx + jj] * filter[m+filter_center];
			}

			im_conv[ii*nx + jj] = pixel;
		}

	// Center
	for (ii=filter_center; ii<ny-filter_center; ii++)
		for (jj=0; jj<nx; jj++)	
		{
			pixel = 0;

			for(m =-filter_center; m<=filter_center; ++m)  // full kernel
			{
				pixel += frame_in[(ii+m)*nx + jj] * filter[m+filter_center];
			}

			im_conv[ii*nx + jj] = pixel;
		}

	for (ii=ny-filter_center; ii<ny; ii++)
		for (jj=0; jj<nx; jj++)	
		{
			pixel = 0;

			for(m =-filter_center; m<=filter_center; ++m)  // full kernel
			{
				if (ii+m<ny)
					pixel += frame_in[(ii+m)*nx + jj] * filter[m+filter_center];
			}

			im_conv[ii*nx + jj] = pixel;
		}
}

/************************************/
/* Lucas-Kanade 1 Step              */
/************************************/
void luca_kanade_1step(float *Vx, float *Vy,
	float *Ix, float *Iy, float *It, 
	int spac_conv_size, int temp_conv_size, int window_size, int nx, int ny)
{
	float sumIx2 =0.0;
	float sumIxIy=0.0;
	float sumIy2 =0.0;
	float sumIxIt=0.0;
	float sumIyIt=0.0;
	int i, j, ii, jj;
	int pixel_id;

	int window_center = (window_size-1)/2;

	int spac_conv_center = (spac_conv_size-1)/2;

	for (i=window_center; i<ny-window_center; i++)
		for (j=window_center; j<nx-window_center; j++)
		{
			sumIx2 =0.0;
			sumIxIy=0.0;
			sumIy2 =0.0;
			sumIxIt=0.0;
			sumIyIt=0.0;

			for (ii=-window_center;ii<=window_center; ii++)
				for (jj=-window_center;jj<=window_center; jj++)
				{
					pixel_id = (i+ii)*nx+(j+jj);
					sumIx2  += Ix[pixel_id]*Ix[pixel_id];
					sumIxIy += Ix[pixel_id]*Iy[pixel_id];
					sumIy2  += Iy[pixel_id]*Iy[pixel_id];
					sumIxIt += Ix[pixel_id]*It[pixel_id];
					sumIyIt += Iy[pixel_id]*It[pixel_id];
				}
			float detA = (sumIx2*sumIy2-sumIxIy*sumIxIy);


			//Luca-Kanade desarrollado el producto vectorial con la inversa 2x2
			if (detA!=0.0){
				Vx[i*nx+j] = 1.0/detA*(sumIy2*(-sumIxIt)     + (-sumIxIy)*(-sumIyIt));
				Vy[i*nx+j] = 1.0/detA*((-sumIxIy)*(-sumIxIt) + sumIx2*(-sumIyIt));
				
			} else {
				Vx[i*nx+j] = 0.0;
				Vy[i*nx+j] = 0.0;
			}
		}
}


/********************************************************************/
/* Return the Velocity Vx,Vy from an input images in window_frames  */
/********************************************************************/
void LucasKanade::lucas_kanade(float *Vx, float *Vy, int iframe, unsigned char *wind_frames, int gpu)
{
	unsigned char* frame = wind_frames + (((iframe+1)%temp_filt_size)*nx*ny);

	if (!gpu) {
		// It
		temp_convolution(iframe, It, wind_frames, nx, ny, filt_t, temp_filt_size);

		// Ix = imfilter(frame, filter_x,'left-top');
		spac_convolution2D_x(Ix, frame, nx, ny, filt_x, spac_filt_size);

		// Iy = imfilter(frame, filter_y,'left-top');
		spac_convolution2D_y(Iy, frame, nx, ny, filt_y, spac_filt_size);

		luca_kanade_1step(Vx, Vy, Ix, Iy, It,
			spac_filt_size, temp_filt_size, window_size, nx, ny);
	} else {
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
/*
	printf("[%d]  ", iframe);
	for (int i=2; i<ny; i+=15)
		printf("V[%d]=%f,%f ", i, Vx[i*nx+ny/2], Vy[i*nx+ny/2]);
	printf("\n");
	printf("[%d]  ", iframe);
	for (int i=2; i<ny; i+=15)
		printf("frame[%d]=%d,%d ", i, frame[i*nx+ny/2], frame[nx*ny+i*nx+ny/2]);
	printf("\n");
*/
}

void LucasKanade::copy_frames_circularbuffer_GPU_wrapper(u_char *frame, int temp_size, int fr, int frame_size){
	copy_frames_circularbuffer_GPU(frame, d_wind_frames, temp_size, fr, frame_size);
}