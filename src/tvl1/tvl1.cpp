#include <cmath>
#include <iostream>
#include <algorithm>

#include "mkl.h"

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

TV_L1::TV_L1(int width, int height, float tau, float lambda, float theta, int nscales,
	float zfactor, int warps, float epsilon, bool verbose) 
{
	_width = width;
	_height = height;
	_tau = tau;
	_lambda = lambda;
	_theta = theta;
	_warps = warps;
	_epsilon = epsilon;
	_verbose = verbose;
	_zfactor = zfactor;

    //Set the number of scales according to the size of the
    //images.  The value N is computed to assure that the smaller
    //images of the pyramid don't have a size smaller than 16x16
	const float N = 1 + std::log(std::hypot(width, height)/16.0) / std::log(1 / zfactor);
	_nscales = (N < nscales) ? N : nscales;

	_u = new float[2 * _width*_height]{0};

	// allocate memory for the pyramid structure
	_I0s = new float*[_nscales];
	_I1s = new float*[_nscales];
	_u1s = new float*[_nscales];
	_u2s = new float*[_nscales];
	_nx  = new int[_nscales];
	_ny  = new int[_nscales];

	for (int i = 0; i < _nscales; i++) {
		_I0s[i] = new float[_width*_height];
		_I1s[i] = new float[_width*_height];
		_u1s[i] = new float[_width*_height];
		_u2s[i] = new float[_width*_height];
	}

	_I1x    = new float[_width*_height];
	_I1y    = new float[_width*_height];
	_I1w    = new float[_width*_height];
	_I1wx   = new float[_width*_height];
	_I1wy   = new float[_width*_height];
	_rho_c  = new float[_width*_height];
	_v1     = new float[_width*_height];
	_v2     = new float[_width*_height];
	_p11    = new float[_width*_height];
	_p12    = new float[_width*_height];
	_p21    = new float[_width*_height];
	_p22    = new float[_width*_height];
	_grad   = new float[_width*_height];
	_div_p1 = new float[_width*_height];
	_div_p2 = new float[_width*_height];
}

TV_L1::~TV_L1() {
	delete[] _u;
	
	for (int i = 0; i < _nscales; i++) {
		delete[] _I0s[i];
		delete[] _I1s[i];
		delete[] _u1s[i];
		delete[] _u2s[i];
	}

	delete[] _I0s;
	delete[] _I1s;
	delete[] _u1s;
	delete[] _u2s;
	delete[] _nx;
	delete[] _ny;

	delete[] _I1x;
	delete[] _I1y;
	delete[] _I1w;
	delete[] _I1wx;
	delete[] _I1wy;
	delete[] _rho_c;
	delete[] _v1;
	delete[] _v2;
	delete[] _p11;
	delete[] _p12;
	delete[] _p21;
	delete[] _p22;
	delete[] _grad;
	delete[] _div_p1;
	delete[] _div_p2;
}


void TV_L1::convertToFloat(int size, const uint8_t *I0, const uint8_t *I1, float *_I0, float* _I1){
	for (size_t i = 0; i < size; i++) {
		_I0[i] = static_cast<float>(I0[i]);
		_I1[i] = static_cast<float>(I1[i]);
	}
}


/**
 * Function to compute the optical flow using multiple scales
 **/
void TV_L1::runDualTVL1Multiscale(uint8_t *I0, uint8_t *I1) {
	const int size = _width * _height;

	_u1s[0] = _u;
	_u2s[0] = _u + _width*_height;
	_nx [0] = _width;
	_ny [0] = _height;

	convertToFloat(size, I0, I1, _I0s[0], _I1s[0]);

	// normalize the images between 0 and 255
	image_normalization(_I0s[0], _I1s[0], _I0s[0], _I1s[0], size);

	// pre-smooth the original images
	gaussian(_I0s[0], _nx[0], _ny[0], PRESMOOTHING_SIGMA, _I1w);
	gaussian(_I1s[0], _nx[0], _ny[0], PRESMOOTHING_SIGMA, _I1w);

	// create the scales
	for (int s = 1; s < _nscales; s++)
	{
		zoom_size(_nx[s-1], _ny[s-1], &_nx[s], &_ny[s], _zfactor);

		// zoom in the images to create the pyramidal structure
		zoom_out(_I0s[s-1], _I0s[s], _nx[s-1], _ny[s-1], _zfactor, _I1w, _I1wx);
		zoom_out(_I1s[s-1], _I1s[s], _nx[s-1], _ny[s-1], _zfactor, _I1w, _I1wx);
	}

	// initialize the flow at the coarsest scale
	for (int i = 0; i < _nx[_nscales-1] * _ny[_nscales-1]; i++)
		_u1s[_nscales-1][i] = _u2s[_nscales-1][i] = 0.0;

	const float invZfactor{1 / _zfactor};
	// pyramidal structure for computing the optical flow
	for (int s = _nscales-1; s > 0; s--) {
		// compute the optical flow at the current scale
		dualTVL1(_I0s[s], _I1s[s], _u1s[s], _u2s[s], _nx[s], _ny[s]);

		// zoom the optical flow for the next finer scale
		zoom_in(_u1s[s], _u1s[s-1], _nx[s], _ny[s], _nx[s-1], _ny[s-1]);
		zoom_in(_u2s[s], _u2s[s-1], _nx[s], _ny[s], _nx[s-1], _ny[s-1]);

		// scale the optical flow with the appropriate zoom factor
		cblas_sscal (_nx[s-1] * _ny[s-1], invZfactor, _u1s[s-1], 1);
		cblas_sscal (_nx[s-1] * _ny[s-1], invZfactor, _u2s[s-1], 1);
	}
	dualTVL1(_I0s[0], _I1s[0], _u1s[0], _u2s[0], _nx[0], _ny[0]);
}

/**
 *
 * Function to compute the optical flow in one scale
 *
 **/
void TV_L1::dualTVL1(const float* I0, const float* I1, float* u1, float* u2, int nx, int ny)
{
	const int size = nx * ny;
	const float l_t = _lambda * _theta;

	centered_gradient(I1, _I1x, _I1y, nx, ny);

	// initialization of p
	std::fill(_p11, _p11 + size, 0.0f);
	std::fill(_p12, _p12 + size, 0.0f);
	std::fill(_p21, _p21 + size, 0.0f);
	std::fill(_p22, _p22 + size, 0.0f);

	for (int warpings = 0; warpings < _warps; warpings++)
	{
		// compute the warping of the target image and its derivatives
		bicubic_interpolation_warp(I1,  u1, u2, _I1w,  nx, ny, true);
		bicubic_interpolation_warp(_I1x, u1, u2, _I1wx, nx, ny, true);
		bicubic_interpolation_warp(_I1y, u1, u2, _I1wy, nx, ny, true);

#pragma omp parallel for
		for (int i = 0; i < size; i++)
		{
			const float Ix2 = _I1wx[i] * _I1wx[i];
			const float Iy2 = _I1wy[i] * _I1wy[i];

			// store the |Grad(I1)|^2
			_grad[i] = (Ix2 + Iy2);

			// compute the constant part of the rho function
			_rho_c[i] = (_I1w[i] - _I1wx[i] * u1[i]
						- _I1wy[i] * u2[i] - I0[i]);
		}

		int n = 0;
		float error = INFINITY;
		while (error > _epsilon * _epsilon && n < MAX_ITERATIONS)
		{
			n++;
			// estimate the values of the variable (v1, v2)
			// (thresholding opterator TH)
			#pragma omp parallel for
			for (int i = 0; i < size; i++)
			{
				const float rho = _rho_c[i]
					+ (_I1wx[i] * u1[i] + _I1wy[i] * u2[i]);

				float d1, d2;

				if (rho < -l_t * _grad[i])
				{
					d1 = l_t * _I1wx[i];
					d2 = l_t * _I1wy[i];
				}
				else
				{
					if (rho > l_t * _grad[i])
					{
						d1 = -l_t * _I1wx[i];
						d2 = -l_t * _I1wy[i];
					}
					else
					{
						if (_grad[i] < GRAD_IS_ZERO)
							d1 = d2 = 0;
						else
						{
							float fi = -rho/_grad[i];
							d1 = fi * _I1wx[i];
							d2 = fi * _I1wy[i];
						}
					}
				}

				_v1[i] = u1[i] + d1;
				_v2[i] = u2[i] + d2;
			}

			// compute the divergence of the dual variable (p1, p2)
			divergence(_p11, _p12, _div_p1, nx ,ny);
			divergence(_p21, _p22, _div_p2, nx ,ny);

			// estimate the values of the optical flow (u1, u2)
			error = 0.0;
			#pragma omp parallel for reduction(+:error)
			for (int i = 0; i < size; i++)
			{
				const float u1k = u1[i];
				const float u2k = u2[i];

				u1[i] = _v1[i] + _theta * _div_p1[i];
				u2[i] = _v2[i] + _theta * _div_p2[i];

				error += (u1[i] - u1k) * (u1[i] - u1k) +
					(u2[i] - u2k) * (u2[i] - u2k);
			}
			error /= size;

			// compute the gradient of the optical flow (Du1, Du2)
			forward_gradient(u1, _div_p1, _v1, nx ,ny);
			forward_gradient(u2, _div_p2, _v2, nx ,ny);

			// estimate the values of the dual variable (p1, p2)
			const float taut = _tau / _theta;
			#pragma omp parallel for
			for (int i = 0; i < size; i++)
			{
				const float g1   = std::hypot(_div_p1[i], _v1[i]);
				const float g2   = std::hypot(_div_p2[i], _v2[i]);
				const float ng1  = 1.0 + taut * g1;
				const float ng2  = 1.0 + taut * g2;

				_p11[i] = (_p11[i] + taut * _div_p1[i]) / ng1;
				_p12[i] = (_p12[i] + taut * _v1[i]) / ng1;
				_p21[i] = (_p21[i] + taut * _div_p2[i]) / ng2;
				_p22[i] = (_p22[i] + taut * _v2[i]) / ng2;
			}
		}

		if (_verbose)
			std::cerr << "Warping: " << warpings << ", Iterations: " << n << ", Error: " << error << std::endl;
	}
}


/**
 *
 * Function to normalize the images between 0 and 255
 *
 **/
void TV_L1::image_normalization(
		const float *I0,  // input image0
		const float *I1,  // input image1
		float *I0n,       // normalized output image0
		float *I1n,       // normalized output image1
		int size          // size of the image
)
{
	// obtain the max and min of each image
	size_t max0 = cblas_isamax(size, I0, 1);
	size_t max1 = cblas_isamax(size, I1, 1);
	size_t min0 = cblas_isamin(size, I0, 1);
	size_t min1 = cblas_isamin(size, I1, 1);	

	// obtain the max and min of both images
	const float max = std::max(I0[max0], I1[max1]);
	const float min = std::min(I0[min0], I1[min1]);
	const float den = max - min;

	if(den <= 0)
		return;

	// normalize both images
	for (int i = 0; i < size; i++) {
		I0n[i] = 255.0 * (I0[i] - min) / den;
		I1n[i] = 255.0 * (I1[i] - min) / den;
	}
}
