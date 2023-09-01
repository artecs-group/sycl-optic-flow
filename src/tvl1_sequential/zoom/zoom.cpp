// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2012, Javier Sánchez Pérez <jsanchez@dis.ulpgc.es>
// All rights reserved.

#include <cmath>

#include "zoom.hpp"
#include "../mask/mask.hpp"
#include "../bicubic_interpolation/bicubic_interpolation.hpp"

/**
  *
  * Downsample an image
  *
**/
void zoom_out(
	const float *I,    // input image
	float *Iout,       // output image
	const int nx,      // image width
	const int ny,      // image height
	const float factor, // zoom factor between 0 and 1
	float* Is,           // temporary working image
	float* gaussBuffer   // Gaussian buffer
)
{
	for(int i = 0; i < nx * ny; i++)
		Is[i] = I[i];

	// compute the size of the zoomed image
	int nxx, nyy;
	zoom_size(nx, ny, &nxx, &nyy, factor);

	// compute the Gaussian sigma for smoothing
	const float sigma = ZOOM_SIGMA_ZERO * std::sqrt(1.0/(factor*factor) - 1.0);

	// pre-smooth the image
	gaussian(Is, nx, ny, sigma, gaussBuffer);

	// re-sample the image using bicubic interpolation
	#pragma omp parallel for
	for (int i1 = 0; i1 < nyy; i1++)
		for (int j1 = 0; j1 < nxx; j1++) {
			const float i2  = (float) i1 / factor;
			const float j2  = (float) j1 / factor;

			float g = bicubic_interpolation_at(Is, j2, i2, nx, ny, false);
			Iout[i1 * nxx + j1] = g;
		}
}


/**
  *
  * Function to upsample the image
  *
**/
void zoom_in(
	const float *I, // input image
	float *Iout,    // output image
	int nx,         // width of the original image
	int ny,         // height of the original image
	int nxx,        // width of the zoomed image
	int nyy         // height of the zoomed image
)
{
	// compute the zoom factor
	const float factorx = ((float)nxx / nx);
	const float factory = ((float)nyy / ny);

	// re-sample the image using bicubic interpolation
	#pragma omp parallel for
	for (int i1 = 0; i1 < nyy; i1++)
	for (int j1 = 0; j1 < nxx; j1++)
	{
		float i2 =  (float) i1 / factory;
		float j2 =  (float) j1 / factorx;

		float g = bicubic_interpolation_at(I, j2, i2, nx, ny, false);
		Iout[i1 * nxx + j1] = g;
	}
}
