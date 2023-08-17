#include <cmath>
#include <iostream>
#include <algorithm>
#include <limits>

#include "tvl1.cuh"
#include "kernels/kernels.cuh"

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
	float zfactor, int warps, float epsilon) 
{
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

	_hostU = new float[2 * _width*_height];
	_hNx   = new int[_nscales];
	_hNy   = new int[_nscales];

	cublasCreate(&_handle);

	// allocate memory for the pyramid structure
	cudaMalloc(&_I0s, _nscales * _width * _height * sizeof(float));
	cudaMalloc(&_I1s, _nscales * _width * _height * sizeof(float));
	cudaMalloc(&_u1s, _nscales * _width * _height * sizeof(float));
	cudaMalloc(&_u2s, _nscales * _width * _height * sizeof(float));
	cudaMalloc(&_nx, _nscales * sizeof(int));
	cudaMalloc(&_ny, _nscales * sizeof(int));
	cudaMalloc(&_nxy, 2 * sizeof(int));

	cudaMalloc(&_I1x, _width*_height * sizeof(float));
	cudaMalloc(&_I1y, _width*_height * sizeof(float));
	cudaMalloc(&_I1w, _width*_height * sizeof(float));
	cudaMalloc(&_I1wx, _width*_height * sizeof(float));
	cudaMalloc(&_I1wy, _width*_height * sizeof(float));
	cudaMalloc(&_rho_c, _width*_height * sizeof(float));
	cudaMalloc(&_v1, _width*_height * sizeof(float));
	cudaMalloc(&_v2, _width*_height * sizeof(float));
	cudaMalloc(&_p11, _width*_height * sizeof(float));
	cudaMalloc(&_p12, _width*_height * sizeof(float));
	cudaMalloc(&_p21, _width*_height * sizeof(float));
	cudaMalloc(&_p22, _width*_height * sizeof(float));
	cudaMalloc(&_grad, _width*_height * sizeof(float));
	cudaMalloc(&_div_p1, _width*_height * sizeof(float));
	cudaMalloc(&_div_p2, _width*_height * sizeof(float));
	cudaMalloc(&_g1, _width*_height * sizeof(float));
	cudaMalloc(&_g2, _width*_height * sizeof(float));
	cudaMalloc(&_error, _width*_height * sizeof(float));

	float sigma = ZOOM_SIGMA_ZERO * std::sqrt(1.0/(_zfactor*_zfactor) - 1.0);
	sigma = std::max(sigma, PRESMOOTHING_SIGMA);
	const int bSize = (int) DEFAULT_GAUSSIAN_WINDOW_SIZE * sigma + 1;
	cudaMalloc(&_B,  bSize * sizeof(float));
}

TV_L1::~TV_L1() {
	delete[] _hostU;
	delete[] _hNx;
	delete[] _hNy;
	cublasDestroy(_handle);

	cudaFree(_I0s);
	cudaFree(_I1s);
	cudaFree(_u1s);
	cudaFree(_u2s);
	cudaFree(_nx);
	cudaFree(_ny);
	cudaFree(_nxy);

	cudaFree(_I1x);
	cudaFree(_I1y);
	cudaFree(_I1w);
	cudaFree(_I1wx);
	cudaFree(_I1wy);
	cudaFree(_rho_c);
	cudaFree(_v1);
	cudaFree(_v2);
	cudaFree(_p11);
	cudaFree(_p12);
	cudaFree(_p21);
	cudaFree(_p22);
	cudaFree(_grad);
	cudaFree(_div_p1);
	cudaFree(_div_p2);
	cudaFree(_g1);
	cudaFree(_g2);
	cudaFree(_B);
	cudaFree(_error);
}


/**
 * Function to compute the optical flow using multiple scales
 **/
void TV_L1::runDualTVL1Multiscale(const float *I0, const float *I1) {
	const int size = _width * _height;

	// send images to the device 
	cudaMemcpy(_I0s, I0, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(_I1s, I1, size * sizeof(float), cudaMemcpyHostToDevice);

	// setup initial values
	cudaMemcpy(_nx, &_width, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(_ny, &_height, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemset(_u1s + (_nscales-1 * size), 0.0f, size * sizeof(float));
	cudaMemset(_u2s + (_nscales-1 * size), 0.0f, size * sizeof(float));

	// normalize the images between 0 and 255
	imageNormalization(_I0s, _I1s, _I0s, _I1s, size);

	// pre-smooth the original images
	try {
		gaussian(_I0s, _B, _nx, _ny, PRESMOOTHING_SIGMA, _I1w, &_handle);
		gaussian(_I1s, _B, _nx, _ny, PRESMOOTHING_SIGMA, _I1w, &_handle);
	}
	catch(const std::exception& e) { throw; }

	// float aux[10];
	// cudaMemcpy(aux, _I0s, 10*sizeof(float), cudaMemcpyDeviceToHost);
	// cudaDeviceSynchronize();
	// for (size_t i = 0; i < 10; i++)
	// 	std::cout << aux[i] << ", ";
	// std::cout << std::endl;
	// return;

	// create the scales
	for (int s = 1; s < _nscales; s++)
	{
		zoomSize<<<1,1>>>(_nx + (s-1), _ny + (s-1), _nx + s, _ny + s, _zfactor);

		// zoom in the images to create the pyramidal structure
		try {
			zoomOut(_I0s + (s-1)*size, _I0s + (s*size), _B, _nx + (s-1), _ny + (s-1), _nxy, _nxy + 1, _zfactor, _I1w, _I1wx, &_handle);
			zoomOut(_I1s + (s-1)*size, _I1s + (s*size), _B, _nx + (s-1), _ny + (s-1), _nxy, _nxy + 1, _zfactor, _I1w, _I1wx, &_handle);
		}
		catch(const std::exception& e) { throw; }
	}
	cudaMemcpy(_hNx, _nx, _nscales * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(_hNy, _ny, _nscales * sizeof(int), cudaMemcpyDeviceToHost);

	const float invZfactor{1 / _zfactor};
	// pyramidal structure for computing the optical flow
	for (int s = _nscales-1; s > 0; s--) {
		// compute the optical flow at the current scale
		dualTVL1(_I0s + (s*size), _I1s + (s*size), _u1s + (s*size), _u2s + (s*size), _hNx[s], _hNy[s]);

		// zoom the optical flow for the next finer scale
		zoomIn(_u1s + (s*size), _u1s + (s-1)*size, _nx + s, _ny + s, _nx + (s-1), _ny + (s-1));
		zoomIn(_u2s + (s*size), _u2s + (s-1)*size, _nx + s, _ny + s, _nx + (s-1), _ny + (s-1));

		// scale the optical flow with the appropriate zoom factor
		cublasSscal(_handle, _hNx[s-1] * _hNy[s-1], &invZfactor, _u1s + (s-1)*size, 1);
		cublasSscal(_handle, _hNx[s-1] * _hNy[s-1], &invZfactor, _u2s + (s-1)*size, 1);
	}
	dualTVL1(_I0s, _I1s, _u1s, _u2s, _hNx[0], _hNy[0]);

	// write back to the host the result
	cudaMemcpy(_hostU, _u1s, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(_hostU + size, _u2s, size * sizeof(float), cudaMemcpyDeviceToHost);
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
	cudaMemset(_p11, 0.0f, size * sizeof(float));
	cudaMemset(_p12, 0.0f, size * sizeof(float));
	cudaMemset(_p21, 0.0f, size * sizeof(float));
	cudaMemset(_p22, 0.0f, size * sizeof(float));

	const size_t TH2{THREADS_PER_BLOCK/4};
	dim3 blocks(nx / TH2 + (nx % TH2 == 0 ? 0:1), ny / TH2 + (ny % TH2 == 0 ? 0:1));
	dim3 threads(blocks.x == 1 ? nx:TH2, blocks.y == 1 ? ny:TH2);

	const size_t blocks1 = size / THREADS_PER_BLOCK + (size % THREADS_PER_BLOCK == 0 ? 0:1);
	const size_t threads1 = blocks1 == 1 ? size:THREADS_PER_BLOCK;

	for (int warpings = 0; warpings < _warps; warpings++) {
		// compute the warping of the target image and its derivatives
		bicubicInterpolationWarp<<<blocks, threads>>>(I1,  u1, u2, _I1w, nx, ny, true);
		bicubicInterpolationWarp<<<blocks, threads>>>(_I1x, u1, u2, _I1wx, nx, ny, true);
		bicubicInterpolationWarp<<<blocks, threads>>>(_I1y, u1, u2, _I1wy, nx, ny, true);

		calculateRhoGrad<<<blocks1,threads1>>>(_I1wx, _I1wy, _I1w, u1, u2, I0, _grad, _rho_c, size);

		int n{0};
		float error{INFINITY};
		while (error > _epsilon * _epsilon && n < MAX_ITERATIONS)
		{
			n++;
			// estimate the values of the variable (v1, v2)
			// (thresholding opterator TH)
			estimateThreshold<<<blocks1,threads1>>>(_rho_c, _I1wx, u1, _I1wy, u2, _grad, lT, size, _v1, _v2);

			// compute the divergence of the dual variable (p1, p2)
			divergence(_p11, _p12, _div_p1, nx ,ny);
			divergence(_p21, _p22, _div_p2, nx ,ny);

			// estimate the values of the optical flow (u1, u2)
			cudaMemset(_error, 0.0f, size * sizeof(float));
			estimateOpticalFlow<<<blocks1,threads1>>>(u1, u2, _v1, _v2, _div_p1, _div_p2, _theta, size, _error);
			cublasSasum(_handle, size, _error, 1, &error);
			error /= size;

			// compute the gradient of the optical flow (Du1, Du2)
			forwardGradient(u1, _div_p1, _v1, nx ,ny);
			forwardGradient(u2, _div_p2, _v2, nx ,ny);

			// estimate the values of the dual variable (p1, p2)
			const float taut = _tau / _theta;
			estimateGArgs<<<blocks1,threads1>>>(_div_p1, _div_p2, _v1, _v2, size, taut, _g1, _g2);

			cublasSaxpy(_handle, size, &taut, _div_p1, 1, _p11, 1);
			cublasSaxpy(_handle, size, &taut, _v1, 1, _p12, 1);
			cublasSaxpy(_handle, size, &taut, _div_p2, 1, _p21, 1);
			cublasSaxpy(_handle, size, &taut, _v2, 1, _p22, 1);

			divideByG<<<blocks1,threads1>>>(_g1, _g2, size, _p11, _p12, _p21, _p22);
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
	int iMax0, iMax1, iMin0, iMin1;
	cublasIsamax(_handle, size, I0, 1, &iMax0);
	cublasIsamax(_handle, size, I1, 1, &iMax1);
	cublasIsamin(_handle, size, I0, 1, &iMin0);
	cublasIsamin(_handle, size, I1, 1, &iMin1);

	// obtain the max and min of both images
	float max0, max1, min0, min1;
	cudaMemcpy(&max0, I0 + iMax0-1, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&max1, I1 + iMax1-1, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&min0, I0 + iMin0-1, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&min1, I1 + iMin1-1, sizeof(float), cudaMemcpyDeviceToHost);

	const float max = std::max(max0, max1);
	const float min = std::min(min0, min1);
	const float den = max - min;

	if(den <= 0)
		return;

	// normalize both images
	const int blocks = size / THREADS_PER_BLOCK + (size % THREADS_PER_BLOCK == 0 ? 0:1);
	const int threads = blocks == 1 ? size : THREADS_PER_BLOCK;
	normKernel<<<blocks, threads>>>(I0, I1, I0n, I1n, min, den, size);
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
	int blocks = (nx-1)*(ny-1) / THREADS_PER_BLOCK + ((nx-1)*(ny-1) % THREADS_PER_BLOCK == 0 ? 0:1);
	int threads = blocks == 1 ? (nx-1)*(ny-1) : THREADS_PER_BLOCK;
	bodyDivergence<<<blocks,threads>>>(v1, v2, div, nx, ny);

	// compute the divergence on the first and last rows
	blocks = (nx-1) / THREADS_PER_BLOCK + ((nx-1) % THREADS_PER_BLOCK == 0 ? 0:1);
	threads = blocks == 1 ? (nx-1) : THREADS_PER_BLOCK;
	edgeRowsDivergence<<<blocks,threads>>>(v1, v2, div, nx, ny);

	// compute the divergence on the first and last columns
	blocks = (ny-1) / THREADS_PER_BLOCK + ((ny-1) % THREADS_PER_BLOCK == 0 ? 0:1);
	threads = blocks == 1 ? (ny-1) : THREADS_PER_BLOCK;
	edgeColumnsDivergence<<<blocks,threads>>>(v1, v2, div, nx, ny);

	cornersDivergence<<<1,1>>>(v1, v2, div, nx, ny);
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
	bodyForwardGradient<<<blocks, threads>>>(f, fx, fy, nx, ny);

	// compute the gradient on the last row
	blocks = (nx-1) / THREADS_PER_BLOCK + ((nx-1) % THREADS_PER_BLOCK == 0 ? 0:1);
	threads = blocks == 1 ? (nx-1) : THREADS_PER_BLOCK;
	rowsForwardGradient<<<blocks, threads>>>(f, fx, fy, nx, ny);

	// compute the gradient on the last column
	blocks = (ny-1) / THREADS_PER_BLOCK + ((ny-1) % THREADS_PER_BLOCK == 0 ? 0:1);
	threads = blocks == 1 ? (ny-1) : THREADS_PER_BLOCK;
	columnsForwardGradient<<<blocks, threads>>>(f, fx, fy, nx, ny);

	// corners
	cudaMemset(fx + (ny * nx - 1), 0.0f, sizeof(float));
	cudaMemset(fy + (ny * nx - 1), 0.0f, sizeof(float));
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


/**
 * In-place Gaussian smoothing of an image
 */
void TV_L1::gaussian(
	float* I,             // input/output image
	float* B,			  // coefficients of the 1D convolution
	const int* xdim,       // image width
	const int* ydim,       // image height
	float sigma,    // Gaussian sigma
	float* buffer,
	cublasHandle_t* handle
)
{
	const float den  = 2*sigma*sigma;
	const float sPi = sigma * std::sqrt(M_PI * 2);
	const int   size = (int) DEFAULT_GAUSSIAN_WINDOW_SIZE * sigma + 1 ;
	int hXdim{0}, hYdim{0};
	cudaMemcpy(&hXdim, xdim, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&hYdim, ydim, sizeof(int), cudaMemcpyDeviceToHost);

	if (size > hXdim) {
		std::cerr << "Gaussian smooth: sigma too large." << std::endl;
		throw;
	}

	// compute the coefficients of the 1D convolution kernel
	int blocks = size / THREADS_PER_BLOCK + (size % THREADS_PER_BLOCK == 0 ? 0:1);
	int threads = (blocks == 1) ? size : THREADS_PER_BLOCK;
	convolution1D<<<blocks, threads>>>(B, size, sPi, den);

	// normalize the 1D convolution kernel
	float norm, hB;
	cublasSasum(*handle, size, B, 1, &norm);
	cudaMemcpy(&hB, B, sizeof(float), cudaMemcpyDeviceToHost);
	norm = 1 / (norm * 2 - hB);
	cublasSscal(*handle, size, &norm, B, 1);

	blocks = hYdim / THREADS_PER_BLOCK + (hYdim % THREADS_PER_BLOCK == 0 ? 0:1);
	threads = (blocks == 1) ? hYdim : THREADS_PER_BLOCK;
	// convolution of each line of the input image
    lineConvolution<<<blocks, threads>>>(I, B, xdim, ydim, size, buffer);

	blocks = hXdim / THREADS_PER_BLOCK + (hXdim % THREADS_PER_BLOCK == 0 ? 0:1);
	threads = (blocks == 1) ? hXdim : THREADS_PER_BLOCK;
	// convolution of each column of the input image
    columnConvolution<<<blocks, threads>>>(I, B, xdim, ydim, size, buffer);
}


/**
 * Downsample an image
**/
void TV_L1::zoomOut(
	const float *I,    // input image
	float* Iout,       // output image
	float* B,
	const int* nx,      // image width
	const int* ny,      // image height
	int* nxx,
	int* nyy,
	const float factor, // zoom factor between 0 and 1
	float* Is,           // temporary working image
	float* gaussBuffer,
	cublasHandle_t* handle
)
{
	int sx, sy;
	cudaMemcpy(&sx, nx, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&sy, ny, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(Is, I, sx*sy * sizeof(float), cudaMemcpyDeviceToDevice);

	// compute the size of the zoomed image
	sx = (int)(sx * factor + 0.5);
	sy = (int)(sy * factor + 0.5);

	cudaMemcpy(nxx, &sx, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(nyy, &sy, sizeof(int), cudaMemcpyHostToDevice);

	// compute the Gaussian sigma for smoothing
	const float sigma = ZOOM_SIGMA_ZERO * std::sqrt(1.0/(factor*factor) - 1.0);

	// pre-smooth the image
	try { gaussian(Is, B, nx, ny, sigma, gaussBuffer, handle); }
	catch(const std::exception& e) { throw; }

	// re-sample the image using bicubic interpolation
	const size_t TH2{THREADS_PER_BLOCK/4};
	dim3 blocks(sx / TH2 + (sx % TH2 == 0 ? 0:1), sy / TH2 + (sy % TH2 == 0 ? 0:1));
	dim3 threads(blocks.x == 1 ? sx:TH2, blocks.y == 1 ? sy:TH2);
	bicubicResample<<<blocks, threads>>>(Is, Iout, nxx, nyy, nx, ny, factor);
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
	cudaMemcpy(&sx, nxx, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&sy, nyy, sizeof(int), cudaMemcpyDeviceToHost);

	// re-sample the image using bicubic interpolation	
	const size_t TH2{THREADS_PER_BLOCK/4};
	dim3 blocks(sx / TH2 + (sx % TH2 == 0 ? 0:1), sy / TH2 + (sy % TH2 == 0 ? 0:1));
	dim3 threads(blocks.x == 1 ? sx:TH2, blocks.y == 1 ? sy:TH2);
	bicubicResample2<<<blocks, threads>>>(I, Iout, nxx, nyy, nx, ny);
}
