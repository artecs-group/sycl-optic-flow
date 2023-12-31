#include "kernels.cuh"
#include <stdio.h>

#define NTHREADS2D 16

void copy_frames_circularbuffer_GPU(unsigned char *raw_frame, unsigned char *d_wind_frames, int temp_conv_size, int fr, int frame_size)
{
	int idx = fr%temp_conv_size;
	cudaMemcpy(d_wind_frames + idx*frame_size, raw_frame, frame_size*sizeof(unsigned char), cudaMemcpyHostToDevice);
}



__global__ void temp_convolution_GPU(int iframe, float *It, unsigned char *frame_in, int nx, int ny, float *filter, int filter_size)
{
	int i, j;

	i = blockIdx.y * blockDim.y + threadIdx.y; 
	j = blockIdx.x * blockDim.x + threadIdx.x; 

	if (i<ny && j<nx)
		It[i*nx+j] = 0.0f;

	for(int fr=0; fr<filter_size; fr++){
		unsigned char *im = frame_in + (((fr+iframe+1)%filter_size)*nx*ny);
		if (i<ny && j<nx){
			It[i*nx+j] += filter[fr]*im[i*nx+j];
		}
	}
}

void temp_convolution_GPU_wrapper(int iframe, float *It, unsigned char *frame_in, int nx, int ny, float *filter, int filter_size)
{
	dim3 nThs(NTHREADS2D, NTHREADS2D);
	int blocksX = nx/NTHREADS2D;
	if (nx%NTHREADS2D>0) blocksX++;
	int blocksY = ny/NTHREADS2D;
	if (ny%NTHREADS2D>0) blocksY++;
	dim3 dimBlock(blocksX, blocksY);

	temp_convolution_GPU<<<dimBlock,nThs>>>(iframe, It, frame_in, nx, ny, filter, filter_size);
}

__global__ void spac_convolution2D_x_GPU(float *im_conv, unsigned char *frame_in, int nx, int ny, float *filter, int filter_size)
{
	int i, j, k;
	int filter_center = (filter_size-1)/2;

	i = blockIdx.y * blockDim.y + threadIdx.y; 
	j = blockIdx.x * blockDim.x + threadIdx.x; 

	if (i<ny && j<nx){
		im_conv[i*nx+j] = 0.0f;
		if (j<filter_center) {
			for (k=filter_center-j; k<filter_size; k++)
			{
				int pos = j+k-filter_center;
				im_conv[i*nx+j] += frame_in[i*nx+pos]*filter[k];
			}
		} else if (j>=nx-filter_center){
			for (k=0; k<filter_center+(nx-j); k++)
			{
				int pos = j+k-filter_center;
				im_conv[i*nx+j] += frame_in[i*nx+pos]*filter[k];

			}
		} else {
			for (k=0; k<filter_size; k++)
			{
				int pos = j+k-filter_center;
				im_conv[i*nx+j] += frame_in[i*nx+pos]*filter[k];
			}
		}
	}	
}



void spac_convolution2D_x_GPU_wrapper(float *Ix, unsigned char *frame, int nx, int ny, float *filter, int filter_size){
	dim3 nThs(NTHREADS2D, NTHREADS2D);
	int blocksX = nx/NTHREADS2D;
	if (nx%NTHREADS2D>0) blocksX++;
	int blocksY = ny/NTHREADS2D;
	if (ny%NTHREADS2D>0) blocksY++;
	dim3 dimBlock(blocksX, blocksY);

	spac_convolution2D_x_GPU<<<dimBlock,nThs>>>(Ix, frame, nx, ny, filter, filter_size);
}

__global__ void spac_convolution2D_y_GPU(float *im_conv, unsigned char *frame_in, int nx, int ny, float *filter, int filter_size)
{
	int i, j, k;
	int filter_center = (filter_size-1)/2;

	i = blockIdx.y * blockDim.y + threadIdx.y; 
	j = blockIdx.x * blockDim.x + threadIdx.x; 

	if (i<ny && j<nx){
		im_conv[i*nx+j] = 0.0f;
		if (i<filter_center) {
			for (k=filter_center-i; k<filter_size; k++)
			{
				int pos = i+k-filter_center;
				im_conv[i*nx+j] += frame_in[pos*nx+j]*filter[k];
			}
		} else if (i>=nx-filter_center){
			for (k=0; k<filter_center+(nx-i); k++)
			{
				int pos = i+k-filter_center;
				im_conv[i*nx+j] += frame_in[pos*nx+j]*filter[k];

			}
		} else {
			for (k=0; k<filter_size; k++)
			{
				int pos = i+k-filter_center;
				im_conv[i*nx+j] += frame_in[pos*nx+j]*filter[k];
			}
		}
	}
}

void spac_convolution2D_y_GPU_wrapper(float *Iy, unsigned char *frame, int nx, int ny, float *filter, int filter_size){
	dim3 nThs(NTHREADS2D, NTHREADS2D);
	int blocksX = nx/NTHREADS2D;
	if (nx%NTHREADS2D>0) blocksX++;
	int blocksY = ny/NTHREADS2D;
	if (ny%NTHREADS2D>0) blocksY++;
	dim3 dimBlock(blocksX, blocksY);

	spac_convolution2D_y_GPU<<<dimBlock,nThs>>>(Iy, frame, nx, ny, filter, filter_size);
}

__global__ void luca_kanade_1step_GPU(float *Vx, float *Vy, float *Ix, float *Iy, float *It, 
	int spac_filt_size, int temp_filt_size, int window_size, int nx, int ny)
{
	float sumIx2 =0.0f;
	float sumIxIy=0.0f;
	float sumIy2 =0.0f;
	float sumIxIt=0.0f;
	float sumIyIt=0.0f;
	int i, j, ii, jj;
	int pixel_id;

	int window_center = (window_size-1)/2;

	i = blockIdx.y * blockDim.y + threadIdx.y; 
	j = blockIdx.x * blockDim.x + threadIdx.x; 

	if (i>=window_center && j>=window_center && i<ny-window_center && j<nx-window_center){

		sumIx2 =0.0f;
		sumIxIy=0.0f;
		sumIy2 =0.0f;
		sumIxIt=0.0f;
		sumIyIt=0.0f;

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
		if (detA!=0.0f){
			Vx[i*nx+j] = 1.0f/detA*(sumIy2*(-sumIxIt)     + (-sumIxIy)*(-sumIyIt));
			Vy[i*nx+j] = 1.0f/detA*((-sumIxIy)*(-sumIxIt) + sumIx2*(-sumIyIt));
			
		} else {
			Vx[i*nx+j] = 0.0f;
			Vy[i*nx+j] = 0.0f;
		}
	}
}



void luca_kanade_1step_GPU_wrapper(float *Vx, float *Vy, 
	float *d_Vx, float *d_Vy, float *Ix, float *Iy, float *It,
	int spac_filt_size, int temp_filt_size, int window_size, int nx, int ny){

	dim3 nThs(NTHREADS2D, NTHREADS2D);
	int blocksX = nx/NTHREADS2D;
	if (nx%NTHREADS2D>0) blocksX++;
	int blocksY = ny/NTHREADS2D;
	if (ny%NTHREADS2D>0) blocksY++;
	dim3 dimBlock(blocksX, blocksY);

	luca_kanade_1step_GPU<<<dimBlock,nThs>>>(d_Vx, d_Vy, Ix, Iy, It, spac_filt_size, temp_filt_size, window_size, nx, ny);

	cudaMemcpy(Vx, d_Vx, nx*ny*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(Vy, d_Vy, nx*ny*sizeof(float), cudaMemcpyDeviceToHost);
}