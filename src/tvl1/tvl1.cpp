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
	_v = _u + _width*_height;

	// allocate memory for the pyramid structure
	_I0s = new float*[_nscales];
	_I1s = new float*[_nscales];
	_u1s = new float*[_nscales];
	_u2s = new float*[_nscales];
	_nx  = new int[_nscales];
	_ny  = new int[_nscales];

	_I0s[0] = new float[_width*_height];
	_I1s[0] = new float[_width*_height];

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
	_div    = new float[_width*_height];
	_grad   = new float[_width*_height];
	_div_p1 = new float[_width*_height];
	_div_p2 = new float[_width*_height];
	_u1x    = new float[_width*_height];
	_u1y    = new float[_width*_height];
	_u2x    = new float[_width*_height];
	_u2y    = new float[_width*_height];
}

TV_L1::~TV_L1() {
	delete[] _u;
	delete[] _I0s[0];
	delete[] _I1s[0];

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
	delete[] _div;
	delete[] _grad;
	delete[] _div_p1;
	delete[] _div_p2;
	delete[] _u1x;
	delete[] _u1y;
	delete[] _u2x;
	delete[] _u2y;
}

/**
 * Function to compute the optical flow using multiple scales
 **/
void TV_L1::runDualTVL1Multiscale(uint8_t *I0, uint8_t *I1) {
	const int size = _width * _height;

	_u1s[0] = _u;
	_u2s[0] = _v;
	_nx [0] = _width;
	_ny [0] = _height;

	// normalize the images between 0 and 255
	image_normalization(I0, I1, _I0s[0], _I1s[0], size);

	// pre-smooth the original images
	gaussian(_I0s[0], _nx[0], _ny[0], PRESMOOTHING_SIGMA);
	gaussian(_I1s[0], _nx[0], _ny[0], PRESMOOTHING_SIGMA);

	// create the scales
	for (int s = 1; s < _nscales; s++)
	{
		zoom_size(_nx[s-1], _ny[s-1], &_nx[s], &_ny[s], _zfactor);
		const int sizes = _nx[s] * _ny[s];

		// allocate memory
		_I0s[s] = new float[sizes];
		_I1s[s] = new float[sizes];
		_u1s[s] = new float[sizes];
		_u2s[s] = new float[sizes];

		// zoom in the images to create the pyramidal structure
		zoom_out(_I0s[s-1], _I0s[s], _nx[s-1], _ny[s-1], _zfactor);
		zoom_out(_I1s[s-1], _I1s[s], _nx[s-1], _ny[s-1], _zfactor);
	}

	// initialize the flow at the coarsest scale
	for (int i = 0; i < _nx[_nscales-1] * _ny[_nscales-1]; i++)
		_u1s[_nscales-1][i] = _u2s[_nscales-1][i] = 0.0;

	// pyramidal structure for computing the optical flow
	for (int s = _nscales-1; s >= 0; s--)
	{
		if (_verbose)
			std::cout << "Scale " << s << ": " << _nx[s] << "x" << _ny[s] << std::endl;

		// compute the optical flow at the current scale
		dualTVL1(_I0s[s], _I1s[s], _u1s[s], _u2s[s], _nx[s], _ny[s]);

		// if this was the last scale, finish now
		if (!s) break;

		// otherwise, upsample the optical flow

		// zoom the optical flow for the next finer scale
		zoom_in(_u1s[s], _u1s[s-1], _nx[s], _ny[s], _nx[s-1], _ny[s-1]);
		zoom_in(_u2s[s], _u2s[s-1], _nx[s], _ny[s], _nx[s-1], _ny[s-1]);

		// scale the optical flow with the appropriate zoom factor
		for (int i = 0; i < _nx[s-1] * _ny[s-1]; i++)
		{
			_u1s[s-1][i] *= (float) 1.0 / _zfactor;
			_u2s[s-1][i] *= (float) 1.0 / _zfactor;
		}
	}

	// delete allocated memory
	for (int i = 1; i < _nscales; i++)
	{
		delete[] _I0s[i];
		delete[] _I1s[i];
		delete[] _u1s[i];
		delete[] _u2s[i];
	}
}

/**
 *
 * Function to compute the optical flow in one scale
 *
 **/
template<typename T>
void TV_L1::dualTVL1(const T* I0, const T* I1, float* u1, float* u2, int nx, int ny)
{
	const int size = nx * ny;
	const float l_t = _lambda * _theta;

	centered_gradient(I1, _I1x, _I1y, nx, ny);

	// initialization of p
	for (int i = 0; i < size; i++)
	{
		_p11[i] = _p12[i] = 0.0;
		_p21[i] = _p22[i] = 0.0;
	}

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

				if (rho < - l_t * _grad[i])
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
			forward_gradient(u1, _u1x, _u1y, nx ,ny);
			forward_gradient(u2, _u2x, _u2y, nx ,ny);

			// estimate the values of the dual variable (p1, p2)
			#pragma omp parallel for
			for (int i = 0; i < size; i++)
			{
				const float taut = _tau / _theta;
				const float g1   = std::hypot(_u1x[i], _u1y[i]);
				const float g2   = std::hypot(_u2x[i], _u2y[i]);
				const float ng1  = 1.0 + taut * g1;
				const float ng2  = 1.0 + taut * g2;

				_p11[i] = (_p11[i] + taut * _u1x[i]) / ng1;
				_p12[i] = (_p12[i] + taut * _u1y[i]) / ng1;
				_p21[i] = (_p21[i] + taut * _u2x[i]) / ng2;
				_p22[i] = (_p22[i] + taut * _u2y[i]) / ng2;
			}
		}

		if (_verbose)
			std::cerr << "Warping: " << warpings << ", Iterations: " << n << ", Error: " << error << std::endl;
	}
}
template void TV_L1::dualTVL1(const u_int8_t* I0, const uint8_t* I1, float* u1, float* u2, int nx, int ny);
template void TV_L1::dualTVL1(const float* I0, const float* I1, float* u1, float* u2, int nx, int ny);

/**
 *
 * Compute the max and min of an array
 *
 **/
void TV_L1::getminmax(
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
void TV_L1::image_normalization(
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
