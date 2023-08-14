// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2012, Javier Sánchez Pérez <jsanchez@dis.ulpgc.es>
// All rights reserved.

#include <cmath>

#include "zoom.cuh"
#include "../mask/mask.cuh"
#include "../bicubic_interpolation/bicubic_interpolation.cuh"


__global__ void bicubicResample(const float* Is, float *Iout, const int* nxx, const int* nyy, 
	const int* nx, const int* ny, float factor){
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
	const float ii = (float)i / factor;
	const float jj = (float)j / factor;

    if (i < *nyy && j < *nxx) {
        Iout[i * *nxx + j] = bicubic_interpolation_at(Is, jj, ii, *nx, *ny, false);
    }
}

__global__ void bicubicResample2(const float* Is, float *Iout, const int* nxx, const int* nyy, 
	const int* nx, const int* ny){
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
	const float ii = (float)i / ((float)*nyy / *ny);
	const float jj = (float)j / ((float)*nxx / *nx);

    if (i < *nyy && j < *nxx) {
        Iout[i * *nxx + j] = bicubic_interpolation_at(Is, jj, ii, *nx, *ny, false);
    }
}


/**
  *
  * Downsample an image
  *
**/
void zoom_out(
	const float *I,    // input image
	float *Iout,       // output image
	float* B,
	const int* nx,      // image width
	const int* ny,      // image height
	int* nxx,
	int* nyy,
	const float factor, // zoom factor between 0 and 1
	float* Is,           // temporary working image
	cublasHandle_t* handle
)
{
	cudaMemcpy(Is, I, *nx * *ny * sizeof(float), cudaMemcpyDeviceToDevice);

	// compute the size of the zoomed image
	zoom_size<<<1,1>>>(nx, ny, nxx, nyy, factor);
	int sx, sy;
	cudaMemcpy(&sx, nxx, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&sy, nyy, sizeof(int), cudaMemcpyDeviceToHost);

	// compute the Gaussian sigma for smoothing
	const float sigma = ZOOM_SIGMA_ZERO * std::sqrt(1.0/(factor*factor) - 1.0);

	// pre-smooth the image
	gaussian(Is, B, nx, ny, sigma, handle);

	// re-sample the image using bicubic interpolation
	const size_t TH2{THREADS_PER_BLOCK/8};
	dim3 blocks(sx / TH2 + (sx % TH2 == 0 ? 0:1), sy / TH2 + (sy % TH2 == 0 ? 0:1));
	dim3 threads(blocks.x == 1 ? sx:TH2, blocks.y == 1 ? sy:TH2);
	bicubicResample<<<blocks, threads>>>(Is, Iout, nxx, nyy, nx, ny, factor);
}


/**
  *
  * Function to upsample the image
  *
**/
void zoom_in(
	const float *I, // input image
	float *Iout,    // output image
	const int* nx,         // width of the original image
	const int* ny,         // height of the original image
	const int* nxx,        // width of the zoomed image
	const int* nyy         // height of the zoomed image
)
{
	int sx, sy;
	cudaMemcpy(&sx, nxx, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&sy, nyy, sizeof(int), cudaMemcpyDeviceToHost);

	// re-sample the image using bicubic interpolation	
	const size_t TH2{THREADS_PER_BLOCK/8};
	dim3 blocks(sx / TH2 + (sx % TH2 == 0 ? 0:1), sy / TH2 + (sy % TH2 == 0 ? 0:1));
	dim3 threads(blocks.x == 1 ? sx:TH2, blocks.y == 1 ? sy:TH2);
	bicubicResample2<<<blocks, threads>>>(I, Iout, nxx, nyy, nx, ny);
}


/**
  *
  * Compute the size of a zoomed image from the zoom factor
  *
**/
__global__ void zoom_size(
	const int* nx,      // width of the orignal image
	const int* ny,      // height of the orignal image
	int* nxx,    // width of the zoomed image
	int* nyy,    // height of the zoomed image
	float factor // zoom factor between 0 and 1
)
{
	//compute the new size corresponding to factor
	//we add 0.5 for rounding off to the closest number
	*nxx = (int)((float) *nx * factor + 0.5);
	*nyy = (int)((float) *ny * factor + 0.5);
}