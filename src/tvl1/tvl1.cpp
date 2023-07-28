
// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2011, Javier Sánchez Pérez <jsanchez@dis.ulpgc.es>
// All rights reserved.

#include <cmath>
#include <iostream>

#include "tvl1.hpp"
#include "mask/mask.hpp"
#include "bicubic_interpolation/bicubic_interpolation.hpp"
#include "zoom/zoom.hpp"

/**
 * Implementation of the Zach, Pock and Bischof dual TV-L1 optic flow method
 *
 * see reference:
 *  [1] C. Zach, T. Pock and H. Bischof, "A Duality Based Approach for Realtime
 *      TV-L1 Optical Flow", In Proceedings of Pattern Recognition (DAGM),
 *      Heidelberg, Germany, pp. 214-223, 2007
 *
 *
 * Details on the total variation minimization scheme can be found in:
 *  [2] A. Chambolle, "An Algorithm for Total Variation Minimization and
 *      Applications", Journal of Mathematical Imaging and Vision, 20: 89-97, 2004
 **/


/**
 *
 * Function to compute the optical flow in one scale
 *
 **/
template<typename T>
void Dual_TVL1_optic_flow(
	T *I0,           // source image
	T *I1,           // target image
	float *u1,           // x component of the optical flow
	float *u2,           // y component of the optical flow
	const int   nx,      // image width
	const int   ny,      // image height
	const float tau,     // time step
	const float lambda,  // weight parameter for the data term
	const float theta,   // weight parameter for (u - v)²
	const int   warps,   // number of warpings per scale
	const float epsilon, // tolerance for numerical convergence
	const bool  verbose  // enable/disable the verbose mode
)
{
	const int   size = nx * ny;
	const float l_t = lambda * theta;

	float *I1x    = new float[size];
	float *I1y    = new float[size];
	float *I1w    = new float[size];
	float *I1wx   = new float[size];
	float *I1wy   = new float[size];
	float *rho_c  = new float[size];
	float *v1     = new float[size];
	float *v2     = new float[size];
	float *p11    = new float[size];
	float *p12    = new float[size];
	float *p21    = new float[size];
	float *p22    = new float[size];
	float *div    = new float[size];
	float *grad   = new float[size];
	float *div_p1 = new float[size];
	float *div_p2 = new float[size];
	float *u1x    = new float[size];
	float *u1y    = new float[size];
	float *u2x    = new float[size];
	float *u2y    = new float[size];

	centered_gradient(I1, I1x, I1y, nx, ny);

	// initialization of p
	for (int i = 0; i < size; i++)
	{
		p11[i] = p12[i] = 0.0;
		p21[i] = p22[i] = 0.0;
	}

	for (int warpings = 0; warpings < warps; warpings++)
	{
		// compute the warping of the target image and its derivatives
		bicubic_interpolation_warp(I1,  u1, u2, I1w,  nx, ny, true);
		bicubic_interpolation_warp(I1x, u1, u2, I1wx, nx, ny, true);
		bicubic_interpolation_warp(I1y, u1, u2, I1wy, nx, ny, true);

#pragma omp parallel for
		for (int i = 0; i < size; i++)
		{
			const float Ix2 = I1wx[i] * I1wx[i];
			const float Iy2 = I1wy[i] * I1wy[i];

			// store the |Grad(I1)|^2
			grad[i] = (Ix2 + Iy2);

			// compute the constant part of the rho function
			rho_c[i] = (I1w[i] - I1wx[i] * u1[i]
						- I1wy[i] * u2[i] - I0[i]);
		}

		int n = 0;
		float error = INFINITY;
		while (error > epsilon * epsilon && n < MAX_ITERATIONS)
		{
			n++;
			// estimate the values of the variable (v1, v2)
			// (thresholding opterator TH)
#pragma omp parallel for
			for (int i = 0; i < size; i++)
			{
				const float rho = rho_c[i]
					+ (I1wx[i] * u1[i] + I1wy[i] * u2[i]);

				float d1, d2;

				if (rho < - l_t * grad[i])
				{
					d1 = l_t * I1wx[i];
					d2 = l_t * I1wy[i];
				}
				else
				{
					if (rho > l_t * grad[i])
					{
						d1 = -l_t * I1wx[i];
						d2 = -l_t * I1wy[i];
					}
					else
					{
						if (grad[i] < GRAD_IS_ZERO)
							d1 = d2 = 0;
						else
						{
							float fi = -rho/grad[i];
							d1 = fi * I1wx[i];
							d2 = fi * I1wy[i];
						}
					}
				}

				v1[i] = u1[i] + d1;
				v2[i] = u2[i] + d2;
			}

			// compute the divergence of the dual variable (p1, p2)
			divergence(p11, p12, div_p1, nx ,ny);
			divergence(p21, p22, div_p2, nx ,ny);

			// estimate the values of the optical flow (u1, u2)
			error = 0.0;
#pragma omp parallel for reduction(+:error)
			for (int i = 0; i < size; i++)
			{
				const float u1k = u1[i];
				const float u2k = u2[i];

				u1[i] = v1[i] + theta * div_p1[i];
				u2[i] = v2[i] + theta * div_p2[i];

				error += (u1[i] - u1k) * (u1[i] - u1k) +
					(u2[i] - u2k) * (u2[i] - u2k);
			}
			error /= size;

			// compute the gradient of the optical flow (Du1, Du2)
			forward_gradient(u1, u1x, u1y, nx ,ny);
			forward_gradient(u2, u2x, u2y, nx ,ny);

			// estimate the values of the dual variable (p1, p2)
#pragma omp parallel for
			for (int i = 0; i < size; i++)
			{
				const float taut = tau / theta;
				const float g1   = hypot(u1x[i], u1y[i]);
				const float g2   = hypot(u2x[i], u2y[i]);
				const float ng1  = 1.0 + taut * g1;
				const float ng2  = 1.0 + taut * g2;

				p11[i] = (p11[i] + taut * u1x[i]) / ng1;
				p12[i] = (p12[i] + taut * u1y[i]) / ng1;
				p21[i] = (p21[i] + taut * u2x[i]) / ng2;
				p22[i] = (p22[i] + taut * u2y[i]) / ng2;
			}
		}

		if (verbose)
			std::cerr << "Warping: " << warpings << ", Iterations: " << n << ", Error: " << error << std::endl;
	}

	// delete allocated memory
	delete[] I1x;
	delete[] I1y;
	delete[] I1w;
	delete[] I1wx;
	delete[] I1wy;
	delete[] rho_c;
	delete[] v1;
	delete[] v2;
	delete[] p11;
	delete[] p12;
	delete[] p21;
	delete[] p22;
	delete[] div;
	delete[] grad;
	delete[] div_p1;
	delete[] div_p2;
	delete[] u1x;
	delete[] u1y;
	delete[] u2x;
	delete[] u2y;
}

template void Dual_TVL1_optic_flow(
	uint8_t *I0,           // source image
	uint8_t *I1,           // target image
	float *u1,           // x component of the optical flow
	float *u2,           // y component of the optical flow
	const int   nx,      // image width
	const int   ny,      // image height
	const float tau,     // time step
	const float lambda,  // weight parameter for the data term
	const float theta,   // weight parameter for (u - v)²
	const int   warps,   // number of warpings per scale
	const float epsilon, // tolerance for numerical convergence
	const bool  verbose  // enable/disable the verbose mode
);

template void Dual_TVL1_optic_flow(
	float *I0,           // source image
	float *I1,           // target image
	float *u1,           // x component of the optical flow
	float *u2,           // y component of the optical flow
	const int   nx,      // image width
	const int   ny,      // image height
	const float tau,     // time step
	const float lambda,  // weight parameter for the data term
	const float theta,   // weight parameter for (u - v)²
	const int   warps,   // number of warpings per scale
	const float epsilon, // tolerance for numerical convergence
	const bool  verbose  // enable/disable the verbose mode
);

/**
 *
 * Compute the max and min of an array
 *
 **/
void getminmax(
	float *min,     // output min
	float *max,     // output max
	const uint8_t *x, // input array
	int n           // array size
)
{
	*min = *max = x[0];
	for (int i = 1; i < n; i++) {
		if (x[i] < *min)
			*min = x[i];
		if (x[i] > *max)
			*max = x[i];
	}
}

/**
 *
 * Function to normalize the images between 0 and 255
 *
 **/
void image_normalization(
		const uint8_t *I0,  // input image0
		const uint8_t *I1,  // input image1
		float *I0n,       // normalized output image0
		float *I1n,       // normalized output image1
		int size          // size of the image
)
{
	float max0, max1, min0, min1;

	// obtain the max and min of each image
	getminmax(&min0, &max0, I0, size);
	getminmax(&min1, &max1, I1, size);

	// obtain the max and min of both images
	const float max = (max0 > max1)? max0 : max1;
	const float min = (min0 < min1)? min0 : min1;
	const float den = max - min;

	if (den > 0)
		// normalize both images
		for (int i = 0; i < size; i++)
		{
			I0n[i] = 255.0 * (I0[i] - min) / den;
			I1n[i] = 255.0 * (I1[i] - min) / den;
		}

	else
		// copy the original images
		for (int i = 0; i < size; i++)
		{
			I0n[i] = I0[i];
			I1n[i] = I1[i];
		}
}


/**
 *
 * Function to compute the optical flow using multiple scales
 *
 **/
void Dual_TVL1_optic_flow_multiscale(
		uint8_t *I0,           // source image
		uint8_t *I1,           // target image
		float *u1,           // x component of the optical flow
		float *u2,           // y component of the optical flow
		const int   nxx,     // image width
		const int   nyy,     // image height
		const float tau,     // time step
		const float lambda,  // weight parameter for the data term
		const float theta,   // weight parameter for (u - v)²
		const int   nscales, // number of scales
		const float zfactor, // factor for building the image piramid
		const int   warps,   // number of warpings per scale
		const float epsilon, // tolerance for numerical convergence
		const bool  verbose  // enable/disable the verbose mode
)
{
	int size = nxx * nyy;

	// allocate memory for the pyramid structure
	float **I0s = new float*[nscales];
	float **I1s = new float*[nscales];
	float **u1s = new float*[nscales];
	float **u2s = new float*[nscales];
	int    *nx  = new int[nscales];
	int    *ny  = new int[nscales];

	I0s[0] = new float[size];
	I1s[0] = new float[size];

	u1s[0] = u1;
	u2s[0] = u2;
	nx [0] = nxx;
	ny [0] = nyy;

	// normalize the images between 0 and 255
	image_normalization(I0, I1, I0s[0], I1s[0], size);

	// pre-smooth the original images
	gaussian(I0s[0], nx[0], ny[0], PRESMOOTHING_SIGMA);
	gaussian(I1s[0], nx[0], ny[0], PRESMOOTHING_SIGMA);

	// create the scales
	for (int s = 1; s < nscales; s++)
	{
		zoom_size(nx[s-1], ny[s-1], &nx[s], &ny[s], zfactor);
		const int sizes = nx[s] * ny[s];

		// allocate memory
		I0s[s] = new float[sizes];
		I1s[s] = new float[sizes];
		u1s[s] = new float[sizes];
		u2s[s] = new float[sizes];

		// zoom in the images to create the pyramidal structure
		zoom_out(I0s[s-1], I0s[s], nx[s-1], ny[s-1], zfactor);
		zoom_out(I1s[s-1], I1s[s], nx[s-1], ny[s-1], zfactor);
	}

	// initialize the flow at the coarsest scale
	for (int i = 0; i < nx[nscales-1] * ny[nscales-1]; i++)
		u1s[nscales-1][i] = u2s[nscales-1][i] = 0.0;

	// pyramidal structure for computing the optical flow
	for (int s = nscales-1; s >= 0; s--)
	{
		if (verbose)
			std::cout << "Scale " << s << ": " << nx[s] << "x" << ny[s] << std::endl;

		// compute the optical flow at the current scale
		Dual_TVL1_optic_flow(
				I0s[s], I1s[s], u1s[s], u2s[s], nx[s], ny[s],
				tau, lambda, theta, warps, epsilon, verbose
		);

		// if this was the last scale, finish now
		if (!s) break;

		// otherwise, upsample the optical flow

		// zoom the optical flow for the next finer scale
		zoom_in(u1s[s], u1s[s-1], nx[s], ny[s], nx[s-1], ny[s-1]);
		zoom_in(u2s[s], u2s[s-1], nx[s], ny[s], nx[s-1], ny[s-1]);

		// scale the optical flow with the appropriate zoom factor
		for (int i = 0; i < nx[s-1] * ny[s-1]; i++)
		{
			u1s[s-1][i] *= (float) 1.0 / zfactor;
			u2s[s-1][i] *= (float) 1.0 / zfactor;
		}
	}

	// delete allocated memory
	for (int i = 1; i < nscales; i++)
	{
		delete[] I0s[i];
		delete[] I1s[i];
		delete[] u1s[i];
		delete[] u2s[i];
	}
	delete[] I0s[0];
	delete[] I1s[0];

	delete[] I0s;
	delete[] I1s;
	delete[] u1s;
	delete[] u2s;
	delete[] nx;
	delete[] ny;
}
