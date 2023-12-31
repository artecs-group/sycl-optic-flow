// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2011, Javier Sánchez Pérez <jsanchez@dis.ulpgc.es>
// All rights reserved.

#include <iostream>
#include <cmath>

#include "mkl.h"

#include "mask.hpp"

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
#pragma omp parallel for schedule(dynamic)
	for (int i = 1; i < ny-1; i++)
	{
		for(int j = 1; j < nx-1; j++)
		{
			const int k = i * nx + j;
			dx[k] = 0.5*(input[k+1] - input[k-1]);
			dy[k] = 0.5*(input[k+nx] - input[k-nx]);
		}
	}

	// compute the gradient on the first and last rows
	for (int j = 1; j < nx-1; j++)
	{
		dx[j] = 0.5*(input[j+1] - input[j-1]);
		dy[j] = 0.5*(input[j+nx] - input[j]);

		const int k = (ny - 1) * nx + j;

		dx[k] = 0.5*(input[k+1] - input[k-1]);
		dy[k] = 0.5*(input[k] - input[k-nx]);
	}

	// compute the gradient on the first and last columns
	for(int i = 1; i < ny-1; i++)
	{
		const int p = i * nx;
		dx[p] = 0.5*(input[p+1] - input[p]);
		dy[p] = 0.5*(input[p+nx] - input[p-nx]);

		const int k = (i+1) * nx - 1;

		dx[k] = 0.5*(input[k] - input[k-1]);
		dy[k] = 0.5*(input[k+nx] - input[k-nx]);
	}

	// compute the gradient at the four corners
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
 * In-place Gaussian smoothing of an image
 *
 */
void gaussian(
	float *I,             // input/output image
	const int xdim,       // image width
	const int ydim,       // image height
	const double sigma,    // Gaussian sigma
	float* buffer		   // Temporary buffer
)
{
	const double den  = 2*sigma*sigma;
	const double sPi = sigma * std::sqrt(M_PI * 2);
	const int   size = (int) (DEFAULT_GAUSSIAN_WINDOW_SIZE * sigma) + 1 ;
	const int   bdx  = xdim + size;
	const int   bdy  = ydim + size;

	if (size > xdim) {
		std::cerr << "GaussianSmooth: sigma too large." << std::endl;
		throw;
	}

	// compute the coefficients of the 1D convolution kernel
	double B[size];
	for(int i = 0; i < size; i++)
		B[i] = 1 / sPi * std::exp(-i * i / den);

	// normalize the 1D convolution kernel
	double norm = cblas_dasum (size, B, 1);
	norm = norm * 2 - B[0];
	cblas_dscal (size, 1/norm, B, 1);

	// convolution of each line of the input image
	for (int k = 0; k < ydim; k++)
	{
		int i, j;
		for (i = size; i < bdx; i++)
			buffer[i] = I[k * xdim + i - size];

		for(i = 0, j = bdx; i < size; i++, j++) {
			buffer[i] = I[k * xdim + size-i];
			buffer[j] = I[k * xdim + xdim-i-1];
		}

		for (i = size; i < bdx; i++)
		{
			double sum = B[0] * buffer[i];
			for (j = 1; j < size; j++ )
				sum += B[j] * ( buffer[i-j] + buffer[i+j] );
			I[k * xdim + i - size] = sum;
		}
	}

	// convolution of each column of the input image
	for (int k = 0; k < xdim; k++)
	{
		int i, j;
		for (i = size; i < bdy; i++)
			buffer[i] = I[(i - size) * xdim + k];

		for (i = 0, j = bdy; i < size; i++, j++) {
			buffer[i] = I[(size-i) * xdim + k];
			buffer[j] = I[(ydim-i-1) * xdim + k];
		}

		for (i = size; i < bdy; i++)
		{
			double sum = B[0] * buffer[i];
			for (j = 1; j < size; j++ )
				sum += B[j] * (buffer[i-j] + buffer[i+j]);
			I[(i - size) * xdim + k] = sum;
		}
	}
}
