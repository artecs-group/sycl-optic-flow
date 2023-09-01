#include <sycl/sycl.hpp>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <limits>

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

TV_L1::TV_L1(sycl::queue queue, int width, int height, float tau, float lambda, float theta, int nscales,
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
	const float N = 1 + std::log(std::hypot(width, height)/16.0) / std::log(1 / zfactor);
	_nscales = (N < nscales) ? N : nscales;

	_hostU = new float[2 * _width*_height]{0};
	_hNx   = new int[_nscales];
	_hNy   = new int[_nscales];

    // allocate memory for the pyramid structure
    _I0s = sycl::malloc_device<float>(_nscales * _width * _height, _queue);
    _I1s = sycl::malloc_device<float>(_nscales * _width * _height, _queue);
    _u1s = sycl::malloc_device<float>(_nscales * _width * _height, _queue);
    _u2s = sycl::malloc_device<float>(_nscales * _width * _height, _queue);
    _imBuffer = sycl::malloc_device<float>(_width * _height, _queue);
    _nx = sycl::malloc_device<int>(_nscales, _queue);
    _ny = sycl::malloc_device<int>(_nscales, _queue);
    _nxy = sycl::malloc_device<int>(2, _queue);

    _I1x = sycl::malloc_device<float>(_width * _height, _queue);
    _I1y = sycl::malloc_device<float>(_width * _height, _queue);
    _I1w = sycl::malloc_device<float>(_width * _height, _queue);
    _I1wx = sycl::malloc_device<float>(_width * _height, _queue);
    _I1wy = sycl::malloc_device<float>(_width * _height, _queue);
    _rho_c = sycl::malloc_device<float>(_width * _height, _queue);
    _v1 = sycl::malloc_device<float>(_width * _height, _queue);
    _v2 = sycl::malloc_device<float>(_width * _height, _queue);
    _p11 = sycl::malloc_device<float>(_width * _height, _queue);
    _p12 = sycl::malloc_device<float>(_width * _height, _queue);
    _p21 = sycl::malloc_device<float>(_width * _height, _queue);
    _p22 = sycl::malloc_device<float>(_width * _height, _queue);
    _grad = sycl::malloc_device<float>(_width * _height, _queue);
    _div_p1 = sycl::malloc_device<float>(_width * _height, _queue);
    _div_p2 = sycl::malloc_device<float>(_width * _height, _queue);
    _g1 = sycl::malloc_device<float>(_width * _height, _queue);
    _g2 = sycl::malloc_device<float>(_width * _height, _queue);
    _error = sycl::malloc_device<float>(_width * _height, _queue);
    _lError = sycl::malloc_shared<float>(1, _queue);
    _maxMin = sycl::malloc_shared<float>(4, _queue);

    float sigma = ZOOM_SIGMA_ZERO * std::sqrt(1.0/(_zfactor*_zfactor) - 1.0);
	sigma = std::max(sigma, PRESMOOTHING_SIGMA);
	const int bSize = (int) DEFAULT_GAUSSIAN_WINDOW_SIZE * sigma + 1;
    _B = sycl::malloc_device<float>(bSize, _queue);
}

TV_L1::~TV_L1() {
    delete[] _hostU;
	delete[] _hNx;
	delete[] _hNy;

    sycl::free(_I0s, _queue);
    sycl::free(_I1s, _queue);
    sycl::free(_u1s, _queue);
    sycl::free(_u2s, _queue);
    sycl::free(_nx, _queue);
    sycl::free(_ny, _queue);
    sycl::free(_nxy, _queue);
    sycl::free(_imBuffer, _queue);

    sycl::free(_I1x, _queue);
    sycl::free(_I1y, _queue);
    sycl::free(_I1w, _queue);
    sycl::free(_I1wx, _queue);
    sycl::free(_I1wy, _queue);
    sycl::free(_rho_c, _queue);
    sycl::free(_v1, _queue);
    sycl::free(_v2, _queue);
    sycl::free(_p11, _queue);
    sycl::free(_p12, _queue);
    sycl::free(_p21, _queue);
    sycl::free(_p22, _queue);
    sycl::free(_grad, _queue);
    sycl::free(_div_p1, _queue);
    sycl::free(_div_p2, _queue);
    sycl::free(_g1, _queue);
    sycl::free(_g2, _queue);
    sycl::free(_B, _queue);
    sycl::free(_error, _queue);
    sycl::free(_lError, _queue);
    sycl::free(_maxMin, _queue);
}


/**
 * Function to compute the optical flow using multiple scales
 **/
void TV_L1::runDualTVL1Multiscale(const float *I) {
    const int size = _width * _height;

	// swap image
    _queue.memcpy(_I0s, _imBuffer, size * sizeof(float));

    // send image to the device
    _queue.memcpy(_imBuffer, I, size * sizeof(float));
    _queue.memcpy(_I1s, _imBuffer, size * sizeof(float));

    // setup initial values
    _queue.memcpy(_nx, &_width, sizeof(int));
    _queue.memcpy(_ny, &_height, sizeof(int));

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
        _queue.submit([&](sycl::handler &cgh) {
            auto _nx_s_ct0 = _nx + (s - 1);
            auto _ny_s_ct1 = _ny + (s - 1);
            auto _nx_s_ct2 = _nx + s;
            auto _ny_s_ct3 = _ny + s;

            cgh.parallel_for(1, [=](sycl::item<1> i) {
                zoomSize(_nx_s_ct0, _ny_s_ct1, _nx_s_ct2, _ny_s_ct3, _zfactor);
            });
        });

        // zoom in the images to create the pyramidal structure
		try {
			zoomOut(_I0s + (s-1)*size, _I0s + (s*size), _B, _nx + (s-1), _ny + (s-1), _nxy, _nxy + 1, _zfactor, _I1w, _I1wx);
			zoomOut(_I1s + (s-1)*size, _I1s + (s*size), _B, _nx + (s-1), _ny + (s-1), _nxy, _nxy + 1, _zfactor, _I1w, _I1wx);
		}
		catch(const std::exception& e) { throw; }
	}

    _queue.memcpy(_hNx, _nx, _nscales * sizeof(int));
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
    _queue.memcpy(_hostU, _u1s, size * sizeof(float)).wait();
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
            oneapi::mkl::blas::column_major::asum(_queue, size, _error, 1, _lError);
            _queue.wait();
            *_lError /= size;

			// compute the gradient of the optical flow (Du1, Du2)
			forwardGradient(u1, _div_p1, _v1, nx ,ny);
			forwardGradient(u2, _div_p2, _v2, nx ,ny);

			// estimate the values of the dual variable (p1, p2)
			const float taut = _tau / _theta;
            estimateGArgs(_div_p1, _div_p2, _v1, _v2, size, taut, _g1, _g2, blocks, threads, _queue);

            oneapi::mkl::blas::column_major::axpy(_queue, size, taut, _div_p1, 1, _p11, 1);
            oneapi::mkl::blas::column_major::axpy(_queue, size, taut, _v1, 1, _p12, 1);
            oneapi::mkl::blas::column_major::axpy(_queue, size, taut, _div_p2, 1, _p21, 1);
            oneapi::mkl::blas::column_major::axpy(_queue, size, taut, _v2, 1, _p22, 1);

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
    oneapi::mkl::blas::column_major::iamin(_queue, size, I1, 1, _maxMin + 3);
    _queue.wait();

    // obtain the max and min of both images
	float max0, max1, min0, min1;
    _queue.memcpy(&max0, I0 + _maxMin[0], sizeof(float));
    _queue.memcpy(&max1, I1 + _maxMin[1], sizeof(float));
    _queue.memcpy(&min0, I0 + _maxMin[2], sizeof(float));
    _queue.memcpy(&min1, I1 + _maxMin[3], sizeof(float));

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
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
        // compute the divergence on the central body of the image
	int blocks = ((nx-1)*(ny-1) - 1) / THREADS_PER_BLOCK + (((nx-1)*(ny-1) - 1) % THREADS_PER_BLOCK == 0 ? 0:1);
	int threads = blocks == 1 ? ((nx-1)*(ny-1) - 1) : THREADS_PER_BLOCK;
        /*
        DPCT1049:9: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
        q_ct1.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                                 sycl::range<3>(1, 1, threads),
                                             sycl::range<3>(1, 1, threads)),
                           [=](sycl::nd_item<3> item_ct1) {
                                   bodyDivergence(v1, v2, div, nx, ny,
                                                  item_ct1);
                           });

        // compute the divergence on the first and last rows
	blocks = (nx-2) / THREADS_PER_BLOCK + ((nx-2) % THREADS_PER_BLOCK == 0 ? 0:1);
	threads = blocks == 1 ? (nx-2) : THREADS_PER_BLOCK;
        /*
        DPCT1049:10: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        q_ct1.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                                 sycl::range<3>(1, 1, threads),
                                             sycl::range<3>(1, 1, threads)),
                           [=](sycl::nd_item<3> item_ct1) {
                                   edgeRowsDivergence(v1, v2, div, nx, ny,
                                                      item_ct1);
                           });

        // compute the divergence on the first and last columns
	blocks = (ny-2) / THREADS_PER_BLOCK + ((ny-2) % THREADS_PER_BLOCK == 0 ? 0:1);
	threads = blocks == 1 ? (ny-2) : THREADS_PER_BLOCK;
        /*
        DPCT1049:11: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        q_ct1.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                                 sycl::range<3>(1, 1, threads),
                                             sycl::range<3>(1, 1, threads)),
                           [=](sycl::nd_item<3> item_ct1) {
                                   edgeColumnsDivergence(v1, v2, div, nx, ny,
                                                         item_ct1);
                           });

        q_ct1.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
            [=](sycl::nd_item<3> item_ct1) {
                    cornersDivergence(v1, v2, div, nx, ny);
            });
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
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
        // compute the gradient on the central body of the image
	int blocks = (nx-1)*(ny-1) / THREADS_PER_BLOCK + ((nx-1)*(ny-1) % THREADS_PER_BLOCK == 0 ? 0:1);
	int threads = blocks == 1 ? (nx-1)*(ny-1) : THREADS_PER_BLOCK;
        /*
        DPCT1049:12: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        q_ct1.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                                 sycl::range<3>(1, 1, threads),
                                             sycl::range<3>(1, 1, threads)),
                           [=](sycl::nd_item<3> item_ct1) {
                                   bodyForwardGradient(f, fx, fy, nx, ny,
                                                       item_ct1);
                           });

        // compute the gradient on the last row
	blocks = (nx-1) / THREADS_PER_BLOCK + ((nx-1) % THREADS_PER_BLOCK == 0 ? 0:1);
	threads = blocks == 1 ? (nx-1) : THREADS_PER_BLOCK;
        /*
        DPCT1049:13: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        q_ct1.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                                 sycl::range<3>(1, 1, threads),
                                             sycl::range<3>(1, 1, threads)),
                           [=](sycl::nd_item<3> item_ct1) {
                                   rowsForwardGradient(f, fx, fy, nx, ny,
                                                       item_ct1);
                           });

        // compute the gradient on the last column
	blocks = (ny-1) / THREADS_PER_BLOCK + ((ny-1) % THREADS_PER_BLOCK == 0 ? 0:1);
	threads = blocks == 1 ? (ny-1) : THREADS_PER_BLOCK;
        /*
        DPCT1049:14: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        q_ct1.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                                 sycl::range<3>(1, 1, threads),
                                             sycl::range<3>(1, 1, threads)),
                           [=](sycl::nd_item<3> item_ct1) {
                                   columnsForwardGradient(f, fx, fy, nx, ny,
                                                          item_ct1);
                           });

        // corners
        q_ct1.memset(fx + (ny * nx - 1), 0.0f, sizeof(float));
        q_ct1.memset(fy + (ny * nx - 1), 0.0f, sizeof(float));
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
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
        // compute the gradient on the center body of the image
	int blocks = ((nx-1)*(ny-1) - 1) / THREADS_PER_BLOCK + (((nx-1)*(ny-1) - 1) % THREADS_PER_BLOCK == 0 ? 0:1);
	int threads = blocks == 1 ? ((nx-1)*(ny-1) - 1) : THREADS_PER_BLOCK;
        /*
        DPCT1049:15: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        q_ct1.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                                 sycl::range<3>(1, 1, threads),
                                             sycl::range<3>(1, 1, threads)),
                           [=](sycl::nd_item<3> item_ct1) {
                                   bodyGradient(input, dx, dy, nx, ny,
                                                item_ct1);
                           });

        // compute the gradient on the first and last rows
	blocks = (nx-2) / THREADS_PER_BLOCK + ((nx-2) % THREADS_PER_BLOCK == 0 ? 0:1);
	threads = blocks == 1 ? (nx-2) : THREADS_PER_BLOCK;
        /*
        DPCT1049:16: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        q_ct1.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                                 sycl::range<3>(1, 1, threads),
                                             sycl::range<3>(1, 1, threads)),
                           [=](sycl::nd_item<3> item_ct1) {
                                   edgeRowsGradient(input, dx, dy, nx, ny,
                                                    item_ct1);
                           });

        // compute the gradient on the first and last columns
	blocks = (ny-2) / THREADS_PER_BLOCK + ((ny-2) % THREADS_PER_BLOCK == 0 ? 0:1);
	threads = blocks == 1 ? (ny-2) : THREADS_PER_BLOCK;
        /*
        DPCT1049:17: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        q_ct1.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                                 sycl::range<3>(1, 1, threads),
                                             sycl::range<3>(1, 1, threads)),
                           [=](sycl::nd_item<3> item_ct1) {
                                   edgeColumnsGradient(input, dx, dy, nx, ny,
                                                       item_ct1);
                           });

        // compute the gradient at the four corners
        q_ct1.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
            [=](sycl::nd_item<3> item_ct1) {
                    cornersGradient(input, dx, dy, nx, ny);
            });
}


/**
 * In-place Gaussian smoothing of an image
 */
void TV_L1::gaussian(float *I,        // input/output image
                     float *B,        // coefficients of the 1D convolution
                     const int *xdim, // image width
                     const int *ydim, // image height
                     float sigma,     // Gaussian sigma
                     float *buffer, dpct::queue_ptr *handle)
{
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
        const float den  = 2*sigma*sigma;
	const float sPi = sigma * std::sqrt(M_PI * 2);
	const int   size = (int) DEFAULT_GAUSSIAN_WINDOW_SIZE * sigma + 1 ;
	int hXdim{0}, hYdim{0};
        q_ct1.memcpy(&hXdim, xdim, sizeof(int));
        q_ct1.memcpy(&hYdim, ydim, sizeof(int));

        if (size > hXdim) {
		std::cerr << "Gaussian smooth: sigma too large." << std::endl;
		throw;
	}

	// compute the coefficients of the 1D convolution kernel
	int blocks = size / THREADS_PER_BLOCK + (size % THREADS_PER_BLOCK == 0 ? 0:1);
	int threads = (blocks == 1) ? size : THREADS_PER_BLOCK;
        /*
        DPCT1049:18: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        q_ct1.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                                 sycl::range<3>(1, 1, threads),
                                             sycl::range<3>(1, 1, threads)),
                           [=](sycl::nd_item<3> item_ct1) {
                                   convolution1D(B, size, sPi, den, item_ct1);
                           });

        // normalize the 1D convolution kernel
	float norm, hB;
        float *res_temp_ptr_ct10 = &norm;
        if (sycl::get_pointer_type(&norm, *handle->get_context()) !=
                sycl::usm::alloc::device &&
            sycl::get_pointer_type(&norm, *handle->get_context()) !=
                sycl::usm::alloc::shared) {
                res_temp_ptr_ct10 =
                    sycl::malloc_shared<float>(1, dpct::get_default_queue());
        }
        oneapi::mkl::blas::column_major::asum(**handle, size, B, 1,
                                              res_temp_ptr_ct10);
        if (sycl::get_pointer_type(&norm, *handle->get_context()) !=
                sycl::usm::alloc::device &&
            sycl::get_pointer_type(&norm, *handle->get_context()) !=
                sycl::usm::alloc::shared) {
                *handle->wait();
                norm = *res_temp_ptr_ct10;
                sycl::free(res_temp_ptr_ct10, dpct::get_default_queue());
        }
        q_ct1.memcpy(&hB, B, sizeof(float));
        norm = 1 / (norm * 2 - hB);
        oneapi::mkl::blas::column_major::scal(**handle, size, norm, B, 1);

        blocks = hYdim / THREADS_PER_BLOCK + (hYdim % THREADS_PER_BLOCK == 0 ? 0:1);
	threads = (blocks == 1) ? hYdim : THREADS_PER_BLOCK;
	// convolution of each line of the input image
    /*
    DPCT1049:19: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
        q_ct1.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                                 sycl::range<3>(1, 1, threads),
                                             sycl::range<3>(1, 1, threads)),
                           [=](sycl::nd_item<3> item_ct1) {
                                   lineConvolution(I, B, xdim, ydim, size,
                                                   buffer, item_ct1);
                           });

        blocks = hXdim / THREADS_PER_BLOCK + (hXdim % THREADS_PER_BLOCK == 0 ? 0:1);
	threads = (blocks == 1) ? hXdim : THREADS_PER_BLOCK;
	// convolution of each column of the input image
    /*
    DPCT1049:20: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
        q_ct1.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                                 sycl::range<3>(1, 1, threads),
                                             sycl::range<3>(1, 1, threads)),
                           [=](sycl::nd_item<3> item_ct1) {
                                   columnConvolution(I, B, xdim, ydim, size,
                                                     buffer, item_ct1);
                           });
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
                    float *gaussBuffer, dpct::queue_ptr *handle)
{
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
        int sx, sy;
        q_ct1.memcpy(&sx, nx, sizeof(int));
        q_ct1.memcpy(&sy, ny, sizeof(int));
        q_ct1.memcpy(Is, I, sx * sy * sizeof(float));

        // compute the size of the zoomed image
	sx = (int)(sx * factor + 0.5);
	sy = (int)(sy * factor + 0.5);

        q_ct1.memcpy(nxx, &sx, sizeof(int));
        q_ct1.memcpy(nyy, &sy, sizeof(int));

        // compute the Gaussian sigma for smoothing
	const float sigma = ZOOM_SIGMA_ZERO * std::sqrt(1.0/(factor*factor) - 1.0);

	// pre-smooth the image
	try { gaussian(Is, B, nx, ny, sigma, gaussBuffer, handle); }
	catch(const std::exception& e) { throw; }

	// re-sample the image using bicubic interpolation
	size_t blocks, threads;
	blocks = (sx*sy) / THREADS_PER_BLOCK + ((sx*sy) % THREADS_PER_BLOCK == 0 ? 0:1);
	threads = (blocks == 1) ? (sx*sy) : THREADS_PER_BLOCK;
        /*
        DPCT1049:21: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        q_ct1.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                                 sycl::range<3>(1, 1, threads),
                                             sycl::range<3>(1, 1, threads)),
                           [=](sycl::nd_item<3> item_ct1) {
                                   bicubicResample(Is, Iout, nxx, nyy, nx, ny,
                                                   factor, item_ct1);
                           });
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
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
        int sx, sy;
        q_ct1.memcpy(&sx, nxx, sizeof(int));
        q_ct1.memcpy(&sy, nyy, sizeof(int));

        // re-sample the image using bicubic interpolation	
	size_t blocks, threads;
	blocks = (sx*sy) / THREADS_PER_BLOCK + ((sx*sy) % THREADS_PER_BLOCK == 0 ? 0:1);
	threads = (blocks == 1) ? (sx*sy) : THREADS_PER_BLOCK;
        /*
        DPCT1049:22: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        q_ct1.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                                 sycl::range<3>(1, 1, threads),
                                             sycl::range<3>(1, 1, threads)),
                           [=](sycl::nd_item<3> item_ct1) {
                                   bicubicResample2(I, Iout, nxx, nyy, nx, ny,
                                                    item_ct1);
                           });
}
