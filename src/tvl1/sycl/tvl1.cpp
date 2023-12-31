#include <cmath>
#include <iostream>
#include <algorithm>
#include <limits>
#include "oneapi/mkl.hpp"

#include "tvl1.hpp"
#include "kernels/kernels.hpp"

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

TV_L1::TV_L1(cl::sycl::queue queue, int width, int height, float tau, float lambda, float theta, int nscales,
	float zfactor, int warps, float epsilon)
{
    _queue = queue;
    _width = width;
	_height = height;
	_tau = tau;
	_lambda = lambda;
	_theta = theta;
	_warps = warps;
	_epsilon = epsilon;
	_zfactor = zfactor;

    //Set the number of scales according to the size of the
    //images.  The value N is computed to assure that the smaller
    //images of the pyramid don't have a size smaller than 16x16
	const float N = 1.0f + std::log(std::hypot(width, height)/16.0f) / std::log(1.0f / zfactor);
	_nscales = (N < nscales) ? N : nscales;

	_hostU = new float[2 * _width*_height]{0.0f};
	_hNx   = new int[_nscales];
	_hNy   = new int[_nscales];

    // allocate memory for the pyramid structure
    _I0s = cl::sycl::malloc_device<float>(_nscales * _width * _height, _queue);
    _I1s = cl::sycl::malloc_device<float>(_nscales * _width * _height, _queue);
    _u1s = cl::sycl::malloc_device<float>(_nscales * _width * _height, _queue);
    _u2s = cl::sycl::malloc_device<float>(_nscales * _width * _height, _queue);
    _imBuffer = cl::sycl::malloc_device<float>(_width * _height, _queue);
    _nx = cl::sycl::malloc_device<int>(_nscales, _queue);
    _ny = cl::sycl::malloc_device<int>(_nscales, _queue);
    _nxy = cl::sycl::malloc_device<int>(2, _queue);

    _I1x = cl::sycl::malloc_device<float>(_width * _height, _queue);
    _I1y = cl::sycl::malloc_device<float>(_width * _height, _queue);
    _I1w = cl::sycl::malloc_device<float>(_width * _height, _queue);
    _I1wx = cl::sycl::malloc_device<float>(_width * _height, _queue);
    _I1wy = cl::sycl::malloc_device<float>(_width * _height, _queue);
    _rho_c = cl::sycl::malloc_device<float>(_width * _height, _queue);
    _v1 = cl::sycl::malloc_device<float>(_width * _height, _queue);
    _v2 = cl::sycl::malloc_device<float>(_width * _height, _queue);
    _p11 = cl::sycl::malloc_device<float>(_width * _height, _queue);
    _p12 = cl::sycl::malloc_device<float>(_width * _height, _queue);
    _p21 = cl::sycl::malloc_device<float>(_width * _height, _queue);
    _p22 = cl::sycl::malloc_device<float>(_width * _height, _queue);
    _grad = cl::sycl::malloc_device<float>(_width * _height, _queue);
    _div_p1 = cl::sycl::malloc_device<float>(_width * _height, _queue);
    _div_p2 = cl::sycl::malloc_device<float>(_width * _height, _queue);
    _g1 = cl::sycl::malloc_device<float>(_width * _height, _queue);
    _g2 = cl::sycl::malloc_device<float>(_width * _height, _queue);
    _error = cl::sycl::malloc_device<float>(_width * _height, _queue);
    _lError = cl::sycl::malloc_shared<float>(1, _queue);
    _maxMin = cl::sycl::malloc_shared<int64_t>(4, _queue);

    float sigma = ZOOM_SIGMA_ZERO * std::sqrt(1.0f/(_zfactor*_zfactor) - 1.0f);
	sigma = std::max(sigma, PRESMOOTHING_SIGMA);
	const int bSize = (int) DEFAULT_GAUSSIAN_WINDOW_SIZE * sigma + 1;
    _B = cl::sycl::malloc_device<float>(bSize, _queue);
}

TV_L1::~TV_L1() {
    delete[] _hostU;
	delete[] _hNx;
	delete[] _hNy;

    cl::sycl::free(_I0s, _queue);
    cl::sycl::free(_I1s, _queue);
    cl::sycl::free(_u1s, _queue);
    cl::sycl::free(_u2s, _queue);
    cl::sycl::free(_nx, _queue);
    cl::sycl::free(_ny, _queue);
    cl::sycl::free(_nxy, _queue);
    cl::sycl::free(_imBuffer, _queue);

    cl::sycl::free(_I1x, _queue);
    cl::sycl::free(_I1y, _queue);
    cl::sycl::free(_I1w, _queue);
    cl::sycl::free(_I1wx, _queue);
    cl::sycl::free(_I1wy, _queue);
    cl::sycl::free(_rho_c, _queue);
    cl::sycl::free(_v1, _queue);
    cl::sycl::free(_v2, _queue);
    cl::sycl::free(_p11, _queue);
    cl::sycl::free(_p12, _queue);
    cl::sycl::free(_p21, _queue);
    cl::sycl::free(_p22, _queue);
    cl::sycl::free(_grad, _queue);
    cl::sycl::free(_div_p1, _queue);
    cl::sycl::free(_div_p2, _queue);
    cl::sycl::free(_g1, _queue);
    cl::sycl::free(_g2, _queue);
    cl::sycl::free(_B, _queue);
    cl::sycl::free(_error, _queue);
    cl::sycl::free(_lError, _queue);
    cl::sycl::free(_maxMin, _queue);
}


/**
 * Function to compute the optical flow using multiple scales
 **/
void TV_L1::runDualTVL1Multiscale(const float *I) {
    const int size = _width * _height;

	// swap image
    _queue.memcpy(_I0s, _imBuffer, size * sizeof(float)).wait();

    // send image to the device
    _queue.memcpy(_imBuffer, I, size * sizeof(float)).wait();
    _queue.memcpy(_I1s, _imBuffer, size * sizeof(float)).wait();

    // setup initial values
    _queue.memcpy(_nx, &_width, sizeof(int)).wait();
    _queue.memcpy(_ny, &_height, sizeof(int)).wait();

    // normalize the images between 0 and 255
	imageNormalization(_I0s, _I1s, _I0s, _I1s, size);

	// pre-smooth the original images
	try {
		gaussian(_I0s, _B, _nx, _ny, PRESMOOTHING_SIGMA, _I1w);
		gaussian(_I1s, _B, _nx, _ny, PRESMOOTHING_SIGMA, _I1w);
	}
	catch(const std::exception& e) { throw; }

	// create the scales
	for (int s = 1; s < _nscales; s++)
	{
        zoomSize(_nx + (s - 1), _ny + (s - 1), _nx + s, _ny + s, _zfactor, _queue);

        // zoom in the images to create the pyramidal structure
		try {
			zoomOut(_I0s + (s-1)*size, _I0s + (s*size), _B, _nx + (s-1), _ny + (s-1), _nxy, _nxy + 1, _zfactor, _I1w, _I1wx);
			zoomOut(_I1s + (s-1)*size, _I1s + (s*size), _B, _nx + (s-1), _ny + (s-1), _nxy, _nxy + 1, _zfactor, _I1w, _I1wx);
		}
		catch(const std::exception& e) { throw; }
	}

    _queue.memcpy(_hNx, _nx, _nscales * sizeof(int)).wait();
    _queue.memcpy(_hNy, _ny, _nscales * sizeof(int)).wait();
    _queue.memset(_u1s + (size * (_nscales - 1)), 0.0f, _hNx[_nscales - 1] * _hNy[_nscales - 1] * sizeof(float));
    _queue.memset(_u2s + (size * (_nscales - 1)), 0.0f, _hNx[_nscales - 1] * _hNy[_nscales - 1] * sizeof(float));

    const float invZfactor{1 / _zfactor};
	// pyramidal structure for computing the optical flow
	for (int s = _nscales-1; s > 0; s--) {
		// compute the optical flow at the current scale
		dualTVL1(_I0s + (s*size), _I1s + (s*size), _u1s + (s*size), _u2s + (s*size), _hNx[s], _hNy[s]);

		// zoom the optical flow for the next finer scale
		zoomIn(_u1s + (s*size), _u1s + (s-1)*size, _nx + s, _ny + s, _nx + (s-1), _ny + (s-1));
		zoomIn(_u2s + (s*size), _u2s + (s-1)*size, _nx + s, _ny + s, _nx + (s-1), _ny + (s-1));

		// scale the optical flow with the appropriate zoom factor
        oneapi::mkl::blas::column_major::scal(_queue, _hNx[s - 1] * _hNy[s - 1], invZfactor, _u1s + (s - 1) * size, 1);
        oneapi::mkl::blas::column_major::scal(_queue, _hNx[s - 1] * _hNy[s - 1], invZfactor, _u2s + (s - 1) * size, 1);
    }
	dualTVL1(_I0s, _I1s, _u1s, _u2s, _hNx[0], _hNy[0]);

	// write back to the host the result
    _queue.memcpy(_hostU, _u1s, size * sizeof(float));
    _queue.memcpy(_hostU + size, _u2s, size * sizeof(float)).wait();
}


/**
 *
 * Function to compute the optical flow in one scale
 *
 **/
void TV_L1::dualTVL1(const float* I0, const float* I1, float* u1, float* u2, int nx, int ny)
{
    const size_t size = nx * ny;
	const float lT = _lambda * _theta;
    const float taut = _tau / _theta;

	centeredGradient(I1, _I1x, _I1y, nx, ny);

	// initialization of p
    _queue.memset(_p11, 0.0f, size * sizeof(float));
    _queue.memset(_p12, 0.0f, size * sizeof(float));
    _queue.memset(_p21, 0.0f, size * sizeof(float));
    _queue.memset(_p22, 0.0f, size * sizeof(float));

    const size_t blocks = size / THREADS_PER_BLOCK + (size % THREADS_PER_BLOCK == 0 ? 0:1);
	const size_t threads = blocks == 1 ? size:THREADS_PER_BLOCK;

	for (int warpings = 0; warpings < _warps; warpings++) {
		// compute the warping of the target image and its derivatives
        bicubicInterpolationWarp(I1, u1, u2, _I1w, nx, ny, true, blocks, threads, _queue);
        bicubicInterpolationWarp(_I1x, u1, u2, _I1wx, nx, ny, true, blocks, threads, _queue);
        bicubicInterpolationWarp(_I1y, u1, u2, _I1wy, nx, ny, true, blocks, threads, _queue);
        _queue.wait();

        calculateRhoGrad(_I1wx, _I1wy, _I1w, u1, u2, I0, _grad, _rho_c, size, blocks, threads, _queue);

        int n{0};
		*_lError = INFINITY;
		while (*_lError > _epsilon * _epsilon && n < MAX_ITERATIONS)
		{
			n++;
			// estimate the values of the variable (v1, v2)
			// (thresholding opterator TH)
            estimateThreshold(_rho_c, _I1wx, u1, _I1wy, u2, _grad, lT, size, _v1, _v2, blocks, threads, _queue);

            // compute the divergence of the dual variable (p1, p2)
			divergence(_p11, _p12, _div_p1, nx ,ny);
			divergence(_p21, _p22, _div_p2, nx ,ny);

			// estimate the values of the optical flow (u1, u2)
            _queue.memset(_error, 0.0f, size * sizeof(float));
            estimateOpticalFlow(u1, u2, _v1, _v2, _div_p1, _div_p2, _theta, size, _error, blocks, threads, _queue);
            oneapi::mkl::blas::column_major::asum(_queue, size, _error, 1, _lError).wait();
            *_lError /= size;

			// compute the gradient of the optical flow (Du1, Du2)
			forwardGradient(u1, _div_p1, _v1, nx ,ny);
			forwardGradient(u2, _div_p2, _v2, nx ,ny);

			// estimate the values of the dual variable (p1, p2)
            estimateGArgs(_div_p1, _div_p2, _v1, _v2, size, taut, _g1, _g2, blocks, threads, _queue);

            oneapi::mkl::blas::column_major::axpy(_queue, size, taut, _div_p1, 1, _p11, 1);
            oneapi::mkl::blas::column_major::axpy(_queue, size, taut, _v1, 1, _p12, 1);
            oneapi::mkl::blas::column_major::axpy(_queue, size, taut, _div_p2, 1, _p21, 1);
            oneapi::mkl::blas::column_major::axpy(_queue, size, taut, _v2, 1, _p22, 1);
            _queue.wait();

            divideByG(_g1, _g2, size, _p11, _p12, _p21, _p22, blocks, threads, _queue);
        }
	}
}


/**
 *
 * Function to normalize the images between 0 and 255
 *
 **/
void TV_L1::imageNormalization(
		const float *I0,  // input image0
		const float *I1,  // input image1
		float *I0n,       // normalized output image0
		float *I1n,       // normalized output image1
		int size          // size of the image
)
{
    // obtain the max and min of each image
    oneapi::mkl::blas::column_major::iamax(_queue, size, I0, 1, _maxMin);
    oneapi::mkl::blas::column_major::iamax(_queue, size, I1, 1, _maxMin + 1);
    oneapi::mkl::blas::column_major::iamin(_queue, size, I0, 1, _maxMin + 2);
    oneapi::mkl::blas::column_major::iamin(_queue, size, I1, 1, _maxMin + 3).wait();

    // obtain the max and min of both images
	float max0, max1, min0, min1;
    _queue.memcpy(&max0, I0 + _maxMin[0], sizeof(float));
    _queue.memcpy(&max1, I1 + _maxMin[1], sizeof(float));
    _queue.memcpy(&min0, I0 + _maxMin[2], sizeof(float));
    _queue.memcpy(&min1, I1 + _maxMin[3], sizeof(float));
    _queue.wait();

    const float max = std::max(max0, max1);
	const float min = std::min(min0, min1);
	const float den = max - min;

	if(den <= 0)
		return;

	// normalize both images
	const int blocks = size / THREADS_PER_BLOCK + (size % THREADS_PER_BLOCK == 0 ? 0:1);
	const int threads = blocks == 1 ? size : THREADS_PER_BLOCK;
    normKernel(I0, I1, I0n, I1n, min, den, size, blocks, threads, _queue);
}


/**
 * Function to compute the divergence with backward differences
 **/
void TV_L1::divergence(
		const float *v1, // x component of the vector field
		const float *v2, // y component of the vector field
		float *div,      // output divergence
		const int nx,    // image width
		const int ny     // image height
)
{
    // compute the divergence on the central body of the image
	int blocks = ((nx-1)*(ny-1) - 1) / THREADS_PER_BLOCK + (((nx-1)*(ny-1) - 1) % THREADS_PER_BLOCK == 0 ? 0:1);
	int threads = blocks == 1 ? ((nx-1)*(ny-1) - 1) : THREADS_PER_BLOCK;
    bodyDivergence(v1, v2, div, nx, ny, blocks, threads, _queue);

    // compute the divergence on the first and last rows
	blocks = (nx-2) / THREADS_PER_BLOCK + ((nx-2) % THREADS_PER_BLOCK == 0 ? 0:1);
	threads = blocks == 1 ? (nx-2) : THREADS_PER_BLOCK;
    edgeRowsDivergence(v1, v2, div, nx, ny, blocks, threads, _queue);

    // compute the divergence on the first and last columns
	blocks = (ny-2) / THREADS_PER_BLOCK + ((ny-2) % THREADS_PER_BLOCK == 0 ? 0:1);
	threads = blocks == 1 ? (ny-2) : THREADS_PER_BLOCK;
    edgeColumnsDivergence(v1, v2, div, nx, ny, blocks, threads, _queue);

    cornersDivergence(v1, v2, div, nx, ny, _queue);
    _queue.wait();
}


/**
 * Function to compute the gradient with forward differences
 **/
void TV_L1::forwardGradient(
		const float *f, //input image
		float *fx,      //computed x derivative
		float *fy,      //computed y derivative
		const int nx,   //image width
		const int ny    //image height
		)
{
    // compute the gradient on the central body of the image
	int blocks = (nx-1)*(ny-1) / THREADS_PER_BLOCK + ((nx-1)*(ny-1) % THREADS_PER_BLOCK == 0 ? 0:1);
	int threads = blocks == 1 ? (nx-1)*(ny-1) : THREADS_PER_BLOCK;
    bodyForwardGradient(f, fx, fy, nx, ny, blocks, threads, _queue);

    // compute the gradient on the last row
	blocks = (nx-1) / THREADS_PER_BLOCK + ((nx-1) % THREADS_PER_BLOCK == 0 ? 0:1);
	threads = blocks == 1 ? (nx-1) : THREADS_PER_BLOCK;
    rowsForwardGradient(f, fx, fy, nx, ny, blocks, threads, _queue);

    // compute the gradient on the last column
	blocks = (ny-1) / THREADS_PER_BLOCK + ((ny-1) % THREADS_PER_BLOCK == 0 ? 0:1);
	threads = blocks == 1 ? (ny-1) : THREADS_PER_BLOCK;
    columnsForwardGradient(f, fx, fy, nx, ny, blocks, threads, _queue);

    // corners
    _queue.memset(fx + (ny * nx - 1), 0.0f, sizeof(float));
    _queue.memset(fy + (ny * nx - 1), 0.0f, sizeof(float));
}


/**
 * Function to compute the gradient with centered differences
 **/
void TV_L1::centeredGradient(
		const float* input,  //input image
		float *dx,           //computed x derivative
		float *dy,           //computed y derivative
		const int nx,        //image width
		const int ny         //image height
		)
{
    // compute the gradient on the center body of the image
	int blocks = ((nx-1)*(ny-1) - 1) / THREADS_PER_BLOCK + (((nx-1)*(ny-1) - 1) % THREADS_PER_BLOCK == 0 ? 0:1);
	int threads = blocks == 1 ? ((nx-1)*(ny-1) - 1) : THREADS_PER_BLOCK;
    bodyGradient(input, dx, dy, nx, ny, blocks, threads, _queue);

    // compute the gradient on the first and last rows
	blocks = (nx-2) / THREADS_PER_BLOCK + ((nx-2) % THREADS_PER_BLOCK == 0 ? 0:1);
	threads = blocks == 1 ? (nx-2) : THREADS_PER_BLOCK;
    edgeRowsGradient(input, dx, dy, nx, ny, blocks, threads, _queue);

    // compute the gradient on the first and last columns
	blocks = (ny-2) / THREADS_PER_BLOCK + ((ny-2) % THREADS_PER_BLOCK == 0 ? 0:1);
	threads = blocks == 1 ? (ny-2) : THREADS_PER_BLOCK;
    edgeColumnsGradient(input, dx, dy, nx, ny, blocks, threads, _queue);

    // compute the gradient at the four corners
    cornersGradient(input, dx, dy, nx, ny, _queue);
}


/**
 * In-place Gaussian smoothing of an image
 */
void TV_L1::gaussian(float *I,        // input/output image
                     float *B,        // coefficients of the 1D convolution
                     const int *xdim, // image width
                     const int *ydim, // image height
                     float sigma,     // Gaussian sigma
                     float *buffer)
{
    const float den  = 2.0f*sigma*sigma;
	const float sPi = sigma * std::sqrt(M_PI * 2.0f);
	const int   size = (int) DEFAULT_GAUSSIAN_WINDOW_SIZE * sigma + 1;
	int hXdim{0}, hYdim{0};
    _queue.memcpy(&hXdim, xdim, sizeof(int));
    _queue.memcpy(&hYdim, ydim, sizeof(int));
    _queue.wait();

    if (size > hXdim) {
        std::cerr << "Gaussian smooth: sigma too large." << std::endl;
        throw;
	}

	// compute the coefficients of the 1D convolution kernel
	int blocks = size / THREADS_PER_BLOCK + (size % THREADS_PER_BLOCK == 0 ? 0:1);
	int threads = (blocks == 1) ? size : THREADS_PER_BLOCK;
    convolution1D(B, size, sPi, den, blocks, threads, _queue);

    // normalize the 1D convolution kernel
	float hB, norm;
    oneapi::mkl::blas::column_major::asum(_queue, size, B, 1, _lError);
    _queue.memcpy(&hB, B, sizeof(float));
    _queue.wait();
    norm = _lError[0];
    norm = 1.0f / (norm * 2.0f - hB);
    oneapi::mkl::blas::column_major::scal(_queue, size, norm, B, 1);

    blocks = hYdim / THREADS_PER_BLOCK + (hYdim % THREADS_PER_BLOCK == 0 ? 0:1);
	threads = (blocks == 1) ? hYdim : THREADS_PER_BLOCK;
	// convolution of each line of the input image
    lineConvolution(I, B, xdim, ydim, size, buffer, blocks, threads, _queue);

    blocks = hXdim / THREADS_PER_BLOCK + (hXdim % THREADS_PER_BLOCK == 0 ? 0:1);
	threads = (blocks == 1) ? hXdim : THREADS_PER_BLOCK;
	// convolution of each column of the input image
    columnConvolution(I, B, xdim, ydim, size, buffer, blocks, threads, _queue);
}


/**
 * Downsample an image
**/
void TV_L1::zoomOut(const float *I, // input image
                    float *Iout,    // output image
                    float *B,
                    const int *nx, // image width
                    const int *ny, // image height
                    int *nxx, int *nyy,
                    const float factor, // zoom factor between 0 and 1
                    float *Is,          // temporary working image
                    float *gaussBuffer)
{
    int sx, sy;
    _queue.memcpy(&sx, nx, sizeof(int));
    _queue.memcpy(&sy, ny, sizeof(int));
    _queue.wait();
    _queue.memcpy(Is, I, sx * sy * sizeof(float));

    // compute the size of the zoomed image
	sx = (int)(sx * factor + 0.5f);
	sy = (int)(sy * factor + 0.5f);

    _queue.memcpy(nxx, &sx, sizeof(int));
    _queue.memcpy(nyy, &sy, sizeof(int));
    _queue.wait();

    // compute the Gaussian sigma for smoothing
	const float sigma = ZOOM_SIGMA_ZERO * std::sqrt(1.0f/(factor*factor) - 1.0f);

	// pre-smooth the image
	try { gaussian(Is, B, nx, ny, sigma, gaussBuffer); }
	catch(const std::exception& e) { throw; }

	// re-sample the image using bicubic interpolation
	size_t blocks, threads;
	blocks = (sx*sy) / THREADS_PER_BLOCK + ((sx*sy) % THREADS_PER_BLOCK == 0 ? 0:1);
	threads = (blocks == 1) ? (sx*sy) : THREADS_PER_BLOCK;
    bicubicResample(Is, Iout, nxx, nyy, nx, ny, factor, blocks, threads, _queue);
}


/**
 * Function to upsample the image
**/
void TV_L1::zoomIn(
	const float *I, // input image
	float *Iout,    // output image
	const int* nx,         // width of the original image
	const int* ny,         // height of the original image
	const int* nxx,        // width of the zoomed image
	const int* nyy         // height of the zoomed image
)
{
    int sx, sy;
    _queue.memcpy(&sx, nxx, sizeof(int));
    _queue.memcpy(&sy, nyy, sizeof(int));
    _queue.wait();

    // re-sample the image using bicubic interpolation	
	size_t blocks, threads;
	blocks = (sx*sy) / THREADS_PER_BLOCK + ((sx*sy) % THREADS_PER_BLOCK == 0 ? 0:1);
	threads = (blocks == 1) ? (sx*sy) : THREADS_PER_BLOCK;
    bicubicResample2(I, Iout, nxx, nyy, nx, ny, blocks, threads, _queue);
}
