// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2011, Javier Sánchez Pérez <jsanchez@dis.ulpgc.es>
// All rights reserved.

#include <iostream>
#include <cmath>

#include "mask.cuh"

/**
 *
 * Details on how to compute the divergence and the grad(u) can be found in:
 * [2] A. Chambolle, "An Algorithm for Total Variation Minimization and
 * Applications", Journal of Mathematical Imaging and Vision, 20: 89-97, 2004
 *
 **/


/**
 *
 * Function to compute the divergence with backward differences
 * (see [2] for details)
 *
 **/
void divergence(
		const float *v1, // x component of the vector field
		const float *v2, // y component of the vector field
		float *div,      // output divergence
		const int nx,    // image width
		const int ny     // image height
	       )
{
	// compute the divergence on the central body of the image
	#pragma omp parallel for schedule(dynamic)
	for (int i = 1; i < ny-1; i++) {
		#pragma omp simd
		#pragma ivdep
		for(int j = 1; j < nx-1; j++) {
			const int p  = i * nx + j;
			div[p]  = (v1[p] - v1[p-1]) + (v2[p] - v2[p-nx]);
		}
	}

	// compute the divergence on the first and last rows
	#pragma omp parallel for simd
	#pragma ivdep
	for (int j = 1; j < nx-1; j++)
	{
		const int p = (ny-1) * nx + j;

		div[j] = v1[j] - v1[j-1] + v2[j];
		div[p] = v1[p] - v1[p-1] - v2[p-nx];
	}

	// compute the divergence on the first and last columns
	#pragma omp parallel for
	for (int i = 1; i < ny-1; i++)
	{
		const int p1 = i * nx;
		const int p2 = (i+1) * nx - 1;

		div[p1] =  v1[p1]   + v2[p1] - v2[p1 - nx];
		div[p2] = -v1[p2-1] + v2[p2] - v2[p2 - nx];

	}

	div[0]         =  v1[0] + v2[0];
	div[nx-1]      = -v1[nx - 2] + v2[nx - 1];
	div[(ny-1)*nx] =  v1[(ny-1)*nx] - v2[(ny-2)*nx];
	div[ny*nx-1]   = -v1[ny*nx - 2] - v2[(ny-1)*nx - 1];
}


/**
 *
 * Function to compute the gradient with forward differences
 * (see [2] for details)
 *
 **/
void forward_gradient(
		const float *f, //input image
		float *fx,      //computed x derivative
		float *fy,      //computed y derivative
		const int nx,   //image width
		const int ny    //image height
		)
{
	// compute the gradient on the central body of the image
	#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < ny-1; i++)
	{
		#pragma omp simd
		#pragma ivdep
		for(int j = 0; j < nx-1; j++)
		{
			const int p  = i * nx + j;
			const int p1 = p + 1;
			const int p2 = p + nx;

			fx[p] = f[p1] - f[p];
			fy[p] = f[p2] - f[p];
		}
	}

	// compute the gradient on the last row
	for (int j = 0; j < nx-1; j++)
	{
		const int p = (ny-1) * nx + j;

		fx[p] = f[p+1] - f[p];
		fy[p] = 0;
	}

	// compute the gradient on the last column
	for (int i = 1; i < ny; i++)
	{
		const int p = i * nx-1;

		fx[p] = 0;
		fy[p] = f[p+nx] - f[p];
	}

	fx[ny * nx - 1] = 0;
	fy[ny * nx - 1] = 0;
}

//// i = 0, 1, 2 ; + 2 = 2, 3, 4
//// j = 0, 1, 2 ; + 2 = 2, 3, 4

__global__ void bodyGradient(const float* input, float* dx, float* dy, int nx, int ny){
	const int i = (blockIdx.x * blockDim.x + threadIdx.x) + nx + 1;
	if(i < (nx-1)*(ny-1)){
		dx[i] = 0.5*(input[i+1] - input[i-1]);
		dy[i] = 0.5*(input[i+nx] - input[i-nx]);
	}
}


__global__ void edgeRowsGradient(const float* input, float* dx, float* dy, int nx, int ny){
	const int j = (blockIdx.x * blockDim.x + threadIdx.x) + 1;
	const int k = (ny - 1) * nx + j;
	if(j < nx-1) {
		dx[j] = 0.5*(input[j+1] - input[j-1]);
		dy[j] = 0.5*(input[j+nx] - input[j]);
		dx[k] = 0.5*(input[k+1] - input[k-1]);
		dy[k] = 0.5*(input[k] - input[k-nx]);
	}
}


__global__ void edgeColumnsGradient(const float* input, float* dx, float* dy, int nx, int ny){
	const int i = (blockIdx.x * blockDim.x + threadIdx.x) + 1;
	const int p = i * nx;
	const int k = (i+1) * nx - 1;
	if(i < ny-1) {
		dx[p] = 0.5*(input[p+1] - input[p]);
		dy[p] = 0.5*(input[p+nx] - input[p-nx]);
		dx[k] = 0.5*(input[k] - input[k-1]);
		dy[k] = 0.5*(input[k+nx] - input[k-nx]);
	}
}


__global__ void cornersGradient(const float* input, float* dx, float* dy, int nx, int ny){
	dx[0] = 0.5*(input[1] - input[0]);
	dy[0] = 0.5*(input[nx] - input[0]);

	dx[nx-1] = 0.5*(input[nx-1] - input[nx-2]);
	dy[nx-1] = 0.5*(input[2*nx-1] - input[nx-1]);

	dx[(ny-1)*nx] = 0.5*(input[(ny-1)*nx + 1] - input[(ny-1)*nx]);
	dy[(ny-1)*nx] = 0.5*(input[(ny-1)*nx] - input[(ny-2)*nx]);

	dx[ny*nx-1] = 0.5*(input[ny*nx-1] - input[ny*nx-1-1]);
	dy[ny*nx-1] = 0.5*(input[ny*nx-1] - input[(ny-1)*nx-1]);
}


/**
 *
 * Function to compute the gradient with centered differences
 *
 **/
void centered_gradient(
		const float* input,  //input image
		float *dx,           //computed x derivative
		float *dy,           //computed y derivative
		const int nx,        //image width
		const int ny         //image height
		)
{
	// compute the gradient on the center body of the image
	int blocks = (nx-1)*(ny-1) / THREADS_PER_BLOCK + ((nx-1)*(ny-1) % THREADS_PER_BLOCK == 0 ? 0:1);
	int threads = blocks == 1 ? (nx-1)*(ny-1) : THREADS_PER_BLOCK;
	bodyGradient<<<blocks,threads>>>(input, dx, dy, nx, ny);

	// compute the gradient on the first and last rows
	blocks = (nx-1) / THREADS_PER_BLOCK + ((nx-1) % THREADS_PER_BLOCK == 0 ? 0:1);
	threads = blocks == 1 ? (nx-1) : THREADS_PER_BLOCK;
	edgeRowsGradient<<<blocks,threads>>>(input, dx, dy, nx, ny);

	// compute the gradient on the first and last columns
	blocks = (ny-1) / THREADS_PER_BLOCK + ((ny-1) % THREADS_PER_BLOCK == 0 ? 0:1);
	threads = blocks == 1 ? (ny-1) : THREADS_PER_BLOCK;
	edgeColumnsGradient<<<blocks,threads>>>(input, dx, dy, nx, ny);

	// compute the gradient at the four corners
	cornersGradient<<<1,1>>>(input, dx, dy, nx, ny);
}


__global__ void convolution1D(float* B, int size, float sPi, float den) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < size)
		B[i] = 1 / sPi * std::exp(-i * i / den);
}


__global__ void lineConvolution(const float *I, float *output, const float *B, int xdim, int ydim, int size, int bdx) {
    int k = blockIdx.x;  // Each block processes a different line (k)
    int i = threadIdx.x + size;  // Current index within the line (including padding)
    int tid = threadIdx.x;

    __shared__ float buffer[THREADS_PER_BLOCK + 2 * MAX_SHARED_SIZE];

    // Load necessary data into shared memory
    if (tid < size) {
        buffer[tid] = I[k * xdim + size - tid];
        buffer[tid + THREADS_PER_BLOCK + size] = I[k * xdim + xdim - tid - 1];
    }

    buffer[tid + size] = I[k * xdim + tid];
    buffer[tid + THREADS_PER_BLOCK] = I[k * xdim + tid + THREADS_PER_BLOCK];

    __syncthreads();

    if (i >= size && i < bdx - size) {
        float sum = B[0] * buffer[tid + size];

        for (int j = 1; j < size; j++) {
            sum += B[j] * (buffer[tid + size - j] + buffer[tid + size + j]);
        }

        output[k * xdim + i - size] = sum;
    }
}


__global__ void columnConvolution(const float* I, float* output, const float* B, int xdim, int ydim, int size, int bdy) {
    int k = blockIdx.x;  // Each block processes a different column (k)
    int i = threadIdx.x + size;  // Current index within the column (including padding)
    int tid = threadIdx.x;

    __shared__ float buffer[THREADS_PER_BLOCK + 2 * MAX_SHARED_SIZE];

    // Load necessary data into shared memory
    if (tid < size) {
        buffer[tid] = I[(tid - size) * xdim + k];
        buffer[tid + THREADS_PER_BLOCK + size] = I[(bdy - tid - 1) * xdim + k];
    }

    buffer[tid + size] = I[tid * xdim + k];
    buffer[tid + THREADS_PER_BLOCK] = I[(tid + THREADS_PER_BLOCK) * xdim + k];

    __syncthreads();

    if (i >= size && i < bdy - size) {
        float sum = B[0] * buffer[tid + size];

        for (int j = 1; j < size; j++) {
            sum += B[j] * (buffer[tid + size - j] + buffer[tid + size + j]);
        }

        output[(i - size) * xdim + k] = sum;
    }
}



/**
 *
 * In-place Gaussian smoothing of an image
 *
 */
void gaussian(
	float* I,             // input/output image
	float* B,			  // coefficients of the 1D convolution
	const int* xdim,       // image width
	const int* ydim,       // image height
	const double sigma,    // Gaussian sigma
	cublasHandle_t* handle
)
{
	const float den  = 2*sigma*sigma;
	const float sPi = sigma * std::sqrt(M_PI * 2);
	const int   size = (int) DEFAULT_GAUSSIAN_WINDOW_SIZE * sigma + 1 ;
	int bdx, bdy, hXdim, hYdim;
	cudaMemcpy(&hXdim, xdim, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&hYdim, ydim, sizeof(int), cudaMemcpyDeviceToHost);
	bdx = hXdim + size;
	bdy = hYdim + size;

	if (size > hXdim) {
		std::cerr << "Gaussian smooth: sigma too large." << std::endl;
		throw;
	}

	// compute the coefficients of the 1D convolution kernel
	int blocks = size / THREADS_PER_BLOCK + (size % THREADS_PER_BLOCK == 0 ? 0:1);
	int threads = blocks == 1 ? size : THREADS_PER_BLOCK;
	convolution1D<<<blocks, THREADS_PER_BLOCK>>>(B, size, sPi, den);

	// normalize the 1D convolution kernel
	float norm, hB;
	cublasSasum(*handle, size, B, 1, &norm);
	cudaMemcpy(&hB, B, sizeof(float), cudaMemcpyDeviceToHost);
	norm = 1 / (norm * 2 - hB);
	cublasSscal(*handle, size, &norm, B, 1);

	// convolution of each line of the input image
    lineConvolution<<<hYdim, THREADS_PER_BLOCK>>>(I, I, B, hXdim, hYdim, size, bdx);

	// convolution of each column of the input image
    columnConvolution<<<hXdim, THREADS_PER_BLOCK>>>(I, I, B, hXdim, hYdim, size, bdy);
}
