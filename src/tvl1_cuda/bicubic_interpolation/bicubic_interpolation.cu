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
int neumann_bc(int x, int nx, bool *out) {
	if(x < 0)
	{
	    x = 0;
	    *out = true;
	}
	else if (x >= nx)
	{
	    x = nx - 1;
	    *out = true;
	}

	return x;
}


/**
  *
  * Cubic interpolation in one dimension
  *
**/
inline float cubic_interpolation_cell (
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
inline float bicubic_interpolation_cell (
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
float bicubic_interpolation_at(
	const float* input, //image to be interpolated
	const float  uu,    //x component of the vector field
	const float  vv,    //y component of the vector field
	const int    nx,    //image width
	const int    ny,    //image height
	bool         border_out //if true, return zero outside the region
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
	const float p11 = input[mx  + nx * my];
	const float p12 = input[x   + nx * my];
	const float p13 = input[dx  + nx * my];
	const float p14 = input[ddx + nx * my];

	const float p21 = input[mx  + nx * y];
	const float p22 = input[x   + nx * y];
	const float p23 = input[dx  + nx * y];
	const float p24 = input[ddx + nx * y];

	const float p31 = input[mx  + nx * dy];
	const float p32 = input[x   + nx * dy];
	const float p33 = input[dx  + nx * dy];
	const float p34 = input[ddx + nx * dy];

	const float p41 = input[mx  + nx * ddy];
	const float p42 = input[x   + nx * ddy];
	const float p43 = input[dx  + nx * ddy];
	const float p44 = input[ddx + nx * ddy];

	//create array
	float v[4];
	const float pol[4][4] = {
		{p11, p21, p31, p41},
		{p12, p22, p32, p42},
		{p13, p23, p33, p43},
		{p14, p24, p34, p44}
	};

	//return interpolation
	return bicubic_interpolation_cell(pol, v, uu-x, vv-y);
}


/**
  *
  * Compute the bicubic interpolation of an image.
  *
**/
void bicubic_interpolation_warp(
	const float* input,     // image to be warped
	const float *u,         // x component of the vector field
	const float *v,         // y component of the vector field
	float       *output,    // image warped with bicubic interpolation
	const int    nx,        // image width
	const int    ny,        // image height
	bool         border_out // if true, put zeros outside the region
)
{
	#pragma omp parallel for
	for(int i = 0; i < ny; i++)
		for(int j = 0; j < nx; j++)
		{
			const int   p  = i * nx + j;
			const float uu = j + u[p];
			const float vv = i + v[p];

			// obtain the bicubic interpolation at position (uu, vv)
			output[p] = bicubic_interpolation_at(input,
					uu, vv, nx, ny, border_out);
		}
}
