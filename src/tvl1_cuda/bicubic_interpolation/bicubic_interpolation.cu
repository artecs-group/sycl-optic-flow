// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2012, Javier Sánchez Pérez <jsanchez@dis.ulpgc.es>
// All rights reserved.

#include <cstdint>
#include "bicubic_interpolation.cuh"

/**
  *
  * Neumann boundary condition test
  *
**/
__device__ inline int neumann_bc(int x, int nx, bool *out) {
	*out = (x < 0) || (x >= nx);
	x = max(x, 0);
	return min(x, nx-1);
}


/**
  *
  * Cubic interpolation in one dimension
  *
**/
__device__ inline float cubic_interpolation_cell (
	const float v[4],  //interpolation points
	float x      //point to be interpolated
)
{
	return  v[1] + 0.5 * x * (v[2] - v[0] +
		x * (2.0 *  v[0] - 5.0 * v[1] + 4.0 * v[2] - v[3] +
		x * (3.0 * (v[1] - v[2]) + v[3] - v[0])));
}


/**
  *
  * Bicubic interpolation in two dimensions
  *
**/
__device__ inline float bicubic_interpolation_cell (
	const float p[4][4], //array containing the interpolation points
	float* v, 
	float x,       //x position to be interpolated
	float y        //y position to be interpolated
)
{
	v[0] = cubic_interpolation_cell(p[0], y);
	v[1] = cubic_interpolation_cell(p[1], y);
	v[2] = cubic_interpolation_cell(p[2], y);
	v[3] = cubic_interpolation_cell(p[3], y);
	return cubic_interpolation_cell(v, x);
}

/**
  *
  * Compute the bicubic interpolation of a point in an image.
  * Detect if the point goes outside the image domain.
  *
**/
__device__ float bicubic_interpolation_at(
	const float* input, //image to be interpolated
	float  uu,    //x component of the vector field
	float  vv,    //y component of the vector field
	int    nx,    //image width
	int    ny,    //image height
	bool   border_out //if true, return zero outside the region
)
{
	const int sx = (uu < 0)? -1: 1;
	const int sy = (vv < 0)? -1: 1;

	int x, y, mx, my, dx, dy, ddx, ddy;
	bool out{false};

	x   = neumann_bc((int) uu, nx, &out);
	y   = neumann_bc((int) vv, ny, &out);
	mx  = neumann_bc((int) uu - sx, nx, &out);
	my  = neumann_bc((int) vv - sx, ny, &out);
	dx  = neumann_bc((int) uu + sx, nx, &out);
	dy  = neumann_bc((int) vv + sy, ny, &out);
	ddx = neumann_bc((int) uu + 2*sx, nx, &out);
	ddy = neumann_bc((int) vv + 2*sy, ny, &out);

	if(out && border_out)
		return 0.0;

	//obtain the interpolation points of the image
	float v[4];
	const float pol[4][4] = {
		{input[mx  + nx * my], input[mx  + nx * y], input[mx  + nx * dy], input[mx  + nx * ddy]},
		{input[x   + nx * my], input[x   + nx * y], input[x   + nx * dy], input[x   + nx * ddy]},
		{input[dx  + nx * my], input[dx  + nx * y], input[dx  + nx * dy], input[dx  + nx * ddy]},
		{input[ddx + nx * my], input[ddx + nx * y], input[ddx + nx * dy], input[ddx + nx * ddy]}
	};

	//return interpolation
	return bicubic_interpolation_cell(pol, v, uu-x, vv-y);
}


/**
  *
  * Compute the bicubic interpolation of an image.
  *
**/
__global__ void bicubic_interpolation_warp(
	const float* input,     // image to be warped
	const float *u,         // x component of the vector field
	const float *v,         // y component of the vector field
	float       *output,    // image warped with bicubic interpolation
	int    nx,        // image width
	int    ny,        // image height
	bool border_out // if true, put zeros outside the region
) 
{
	const int j = blockIdx.x * blockDim.x + threadIdx.x;
	const int i = blockIdx.y * blockDim.y + threadIdx.y;
	const int p = i * nx + j;
	const float uu = j + u[p];
	const float vv = i + v[p];

	// obtain the bicubic interpolation at position (uu, vv)
	if(p < nx*ny)
		output[p] = bicubic_interpolation_at(input, uu, vv, nx, ny, border_out);
}
