#include <iostream>
#include <cmath>

#include "kernels.cuh"


__global__ void bodyDivergence(const __half2* v1, const __half2* v2, __half2* div, int nx, int ny){
	const int i = (blockIdx.x * blockDim.x + threadIdx.x) + 1;
	if(i < (nx-1)*(ny-1)){
		div[i]  = (v1[i] - v1[i-1]) + (v2[i] - v2[i-nx]);
	}
}


__global__ void edgeRowsDivergence(const __half2* v1, const __half2* v2, __half2* div, int nx, int ny){
	const int j = (blockIdx.x * blockDim.x + threadIdx.x) + 1;
	const int p = (ny-1) * nx + j;

	if(j < (nx-1)){
		div[j] = v1[j] - v1[j-1] + v2[j];
		div[p] = v1[p] - v1[p-1] - v2[p-nx];
	}
}


__global__ void edgeColumnsDivergence(const __half2* v1, const __half2* v2, __half2* div, int nx, int ny){
	const int i = (blockIdx.x * blockDim.x + threadIdx.x) + 1;
	const int p1 = i * nx;
	const int p2 = (i+1) * nx - 1;

	if(i < (ny-1)){
		div[p1] =  v1[p1]   + v2[p1] - v2[p1 - nx];
		div[p2] = -v1[p2-1] + v2[p2] - v2[p2 - nx];
	}
}


__global__ void cornersDivergence(const __half2* v1, const __half2* v2, __half2* div, int nx, int ny){
	div[0]         =  v1[0] + v2[0];
	div[nx-1]      = -v1[nx - 2] + v2[nx - 1];
	div[(ny-1)*nx] =  v1[(ny-1)*nx] - v2[(ny-2)*nx];
	div[ny*nx-1]   = -v1[ny*nx - 2] - v2[(ny-1)*nx - 1];
}


__global__ void bodyForwardGradient(const __half2* f, __half2* fx, __half2* fy, size_t nx, size_t ny){
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < (nx-1)*(ny-1)){
		fx[i] = f[i+1] - f[i];
		fy[i] = f[i+nx] - f[i];
	}
}


__global__ void rowsForwardGradient(const __half2* f, __half2* fx, __half2* fy, size_t nx, size_t ny){
	const int j = blockIdx.x * blockDim.x + threadIdx.x;
	const int p = (ny-1) * nx + j;

	if(j < (nx-1)){
		fx[p] = f[p+1] - f[p];
		fy[p] = __float2half2_rn(0.0f);
	}
}


__global__ void columnsForwardGradient(const __half2* f, __half2* fx, __half2* fy, size_t nx, size_t ny){
	const int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
	const int p = i * nx-1;

	if(i < ny){
		fx[p] = __float2half2_rn(0.0);
		fy[p] = f[p+nx] - f[p];
	}
}


__global__ void bodyGradient(const __half2* input, __half2* dx, __half2* dy, int nx, int ny){
	const int i = (blockIdx.x * blockDim.x + threadIdx.x) + 1;
	if(i < (nx-1)*(ny-1)){
		dx[i] = __float2half2_rn(0.5) * (input[i+1] - input[i-1]);
		dy[i] = __float2half2_rn(0.5) * (input[i+nx] - input[i-nx]);
	}
}


__global__ void edgeRowsGradient(const __half2* input, __half2* dx, __half2* dy, int nx, int ny){
	const int j = (blockIdx.x * blockDim.x + threadIdx.x) + 1;
	const int k = (ny - 1) * nx + j;
	if(j < nx-1) {
		dx[j] = __float2half2_rn(0.5)*(input[j+1] - input[j-1]);
		dy[j] = __float2half2_rn(0.5)*(input[j+nx] - input[j]);
		dx[k] = __float2half2_rn(0.5)*(input[k+1] - input[k-1]);
		dy[k] = __float2half2_rn(0.5)*(input[k] - input[k-nx]);
	}
}


__global__ void edgeColumnsGradient(const __half2* input, __half2* dx, __half2* dy, int nx, int ny){
	const int i = (blockIdx.x * blockDim.x + threadIdx.x) + 1;
	const int p = i * nx;
	const int k = (i+1) * nx - 1;
	if(i < ny-1) {
		dx[p] = __float2half2_rn(0.5)*(input[p+1] - input[p]);
		dy[p] = __float2half2_rn(0.5)*(input[p+nx] - input[p-nx]);
		dx[k] = __float2half2_rn(0.5)*(input[k] - input[k-1]);
		dy[k] = __float2half2_rn(0.5)*(input[k+nx] - input[k-nx]);
	}
}


__global__ void cornersGradient(const __half2* input, __half2* dx, __half2* dy, int nx, int ny){
	dx[0] = __float2half2_rn(0.5)*(input[1] - input[0]);
	dy[0] = __float2half2_rn(0.5)*(input[nx] - input[0]);

	dx[nx-1] = __float2half2_rn(0.5)*(input[nx-1] - input[nx-2]);
	dy[nx-1] = __float2half2_rn(0.5)*(input[2*nx-1] - input[nx-1]);

	dx[(ny-1)*nx] = __float2half2_rn(0.5)*(input[(ny-1)*nx + 1] - input[(ny-1)*nx]);
	dy[(ny-1)*nx] = __float2half2_rn(0.5)*(input[(ny-1)*nx] - input[(ny-2)*nx]);

	dx[ny*nx-1] = __float2half2_rn(0.5)*(input[ny*nx-1] - input[ny*nx-1-1]);
	dy[ny*nx-1] = __float2half2_rn(0.5)*(input[ny*nx-1] - input[(ny-1)*nx-1]);
}


__global__ void convolution1D(float* B, int size, float sPi, float den) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < size)
		B[i] = 1.0 / sPi * expf((float)-i * i / den);
}


__global__ void lineConvolution(__half2 *I, const float *B, const int* xDim, const int* yDim, int size, __half2* buffer) {
	int k = blockIdx.y * blockDim.y + threadIdx.y; // Row index
	const int xdim{xDim[0]}, ydim{yDim[0]};
    const int bdx = xdim + size;

    if (k < ydim) {
        int i, j;
        for (i = size; i < bdx; i++)
            buffer[i] = I[k * xdim + i - size];

        for (i = 0, j = bdx; i < size; i++, j++) {
            buffer[i] = I[k * xdim + size - i];
            buffer[j] = I[k * xdim + xdim - i - 1];
        }
		const __half2 fB = __float2half2_rn(B[0]);
        for (i = size; i < bdx; i++) {
            __half2 sum = fB * buffer[i];
            for (j = 1; j < size; j++)
                sum += __float2half2_rn(B[j]) * (buffer[i - j] + buffer[i + j]);
            I[k * xdim + i - size] = sum;
        }
    }
}


__global__ void columnConvolution(__half2* I, const float* B, const int* xDim, const int* yDim, int size, __half2* buffer) {
    int k = blockIdx.y * blockDim.y + threadIdx.y; // Row index
	const int xdim{xDim[0]}, ydim{yDim[0]};
    const int bdy = ydim + size;

	if (k < xdim) {
        int i, j;
        for (i = size; i < bdy; i++)
            buffer[i] = I[(i - size) * xdim + k];

        for (i = 0, j = bdy; i < size; i++, j++) {
            buffer[i] = I[(size - i) * xdim + k];
            buffer[j] = I[(ydim - i - 1) * xdim + k];
        }

		const __half2 fB = __float2half2_rn(B[0]);
        for (i = size; i < bdy; i++) {
            __half2 sum = fB * buffer[i];
            for (j = 1; j < size; j++)
                sum += __float2half2_rn(B[j]) * (buffer[i - j] + buffer[i + j]);
            I[(i - size) * xdim + k] = sum;
        }
    }
}


__global__ void bicubicResample(const __half2* Is, __half2 *Iout, const int* nxx, const int* nyy, 
	const int* nx, const int* ny, __half2 factor){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const __half2 ii = __float2half2_rn(static_cast<float>(i)) / factor;
	const __half2 jj = __float2half2_rn(static_cast<float>(j)) / factor;

    if (i < *nyy && j < *nxx) {
        Iout[i * *nxx + j] = bicubicInterpolationAt(Is, jj, ii, *nx, *ny, false);
    }
}


__global__ void bicubicResample2(const __half2* Is, __half2 *Iout, const int* nxx, const int* nyy, 
	const int* nx, const int* ny){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const __half2 ii = __float2half2_rn(i / (*nyy / *ny));
	const __half2 jj = __float2half2_rn(j / (*nxx / *nx));

    if (i < *nyy && j < *nxx) {
        Iout[i * *nxx + j] = bicubicInterpolationAt(Is, jj, ii, *nx, *ny, false);
    }
}


/**
 * Compute the size of a zoomed image from the zoom factor
**/
__global__ void zoomSize(
	const int* nx,      // width of the orignal image
	const int* ny,      // height of the orignal image
	int* nxx,    // width of the zoomed image
	int* nyy,    // height of the zoomed image
	__half2 factor // zoom factor between 0 and 1
)
{
	//compute the new size corresponding to factor
	//we add 0.5 for rounding off to the closest number
	*nxx = (int)(*nx * __high2float(factor) + 0.5);
	*nyy = (int)(*ny * __high2float(factor) + 0.5);
}


/**
 * Neumann boundary condition test
**/
__device__ inline int neumann_bc(int x, int nx, bool* out) {
	*out = (x < 0) || (x >= nx);
	x = max(x, 0);
	return min(x, nx-1);
}


/**
 * Cubic interpolation in one dimension
**/
__device__ inline __half2 cubic_interpolation_cell (
	const __half2 v[4],  //interpolation points
	__half2 x      //point to be interpolated
)
{
	return  v[1] + __float2half2_rn(0.5) * x * (v[2] - v[0] +
		x * (__float2half2_rn(2.0) *  v[0] - __float2half2_rn(5.0) * v[1] + __float2half2_rn(4.0) * v[2] - v[3] +
		x * (__float2half2_rn(3.0) * (v[1] - v[2]) + v[3] - v[0])));
}


/**
 * Bicubic interpolation in two dimensions
**/
__device__ inline __half2 bicubic_interpolation_cell (
	const __half2 p[4][4], //array containing the interpolation points
	__half2* v, 
	__half2 x,       //x position to be interpolated
	__half2 y        //y position to be interpolated
)
{
	v[0] = cubic_interpolation_cell(p[0], y);
	v[1] = cubic_interpolation_cell(p[1], y);
	v[2] = cubic_interpolation_cell(p[2], y);
	v[3] = cubic_interpolation_cell(p[3], y);
	return cubic_interpolation_cell(v, x);
}

/**
  * Compute the bicubic interpolation of a point in an image.
  * Detect if the point goes outside the image domain.
**/
__device__ __half2 bicubicInterpolationAt(
	const __half2* input, //image to be interpolated
	__half2  uu,    //x component of the vector field
	__half2  vv,    //y component of the vector field
	int    nx,    //image width
	int    ny,    //image height
	bool   border_out //if true, return zero outside the region
)
{
	const int sx = (uu < __float2half2_rn(0.0))? -1: 1;
	const int sy = (vv < __float2half2_rn(0.0))? -1: 1;

	int x, y, mx, my, dx, dy, ddx, ddy;
	bool out{false};

	x   = neumann_bc((int) __high2float(uu), nx, &out);
	y   = neumann_bc((int) __high2float(vv), ny, &out);
	mx  = neumann_bc((int) __high2float(uu) - sx, nx, &out);
	my  = neumann_bc((int) __high2float(vv) - sx, ny, &out);
	dx  = neumann_bc((int) __high2float(uu) + sx, nx, &out);
	dy  = neumann_bc((int) __high2float(vv) + sy, ny, &out);
	ddx = neumann_bc((int) __high2float(uu) + 2*sx, nx, &out);
	ddy = neumann_bc((int) __high2float(vv) + 2*sy, ny, &out);

	if(out && border_out)
		return __float2half2_rn(0.0);

	//obtain the interpolation points of the image
	__half2 v[4];
	const __half2 pol[4][4] = {
		{input[mx  + nx * my], input[mx  + nx * y], input[mx  + nx * dy], input[mx  + nx * ddy]},
		{input[x   + nx * my], input[x   + nx * y], input[x   + nx * dy], input[x   + nx * ddy]},
		{input[dx  + nx * my], input[dx  + nx * y], input[dx  + nx * dy], input[dx  + nx * ddy]},
		{input[ddx + nx * my], input[ddx + nx * y], input[ddx + nx * dy], input[ddx + nx * ddy]}
	};

	//return interpolation
	return bicubic_interpolation_cell(pol, v, uu - __float2half2_rn((float)x), vv - __float2half2_rn((float)y));
}


/**
  * Compute the bicubic interpolation of an image.
**/
__global__ void bicubicInterpolationWarp(
	const __half2* input,     // image to be warped
	const __half2 *u,         // x component of the vector field
	const __half2 *v,         // y component of the vector field
	__half2       *output,    // image warped with bicubic interpolation
	int    nx,        // image width
	int    ny,        // image height
	bool border_out // if true, put zeros outside the region
)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int p = i * nx + j;

	// obtain the bicubic interpolation at position (uu, vv)
	if(p < nx*ny){
		const __half2 uu = __float2half2_rn((float)j) + u[p];
		const __half2 vv = __float2half2_rn((float)i) + v[p];
		output[p] = bicubicInterpolationAt(input, uu, vv, nx, ny, border_out);
	}
}


__global__ void calculateRhoGrad(const __half2* I1wx, const __half2* I1wy, const __half2* I1w,
	const __half2* u1, const __half2* u2, const __half2* I0, __half2* grad, __half2* rho_c, int size)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < size) {
		// store the |Grad(I1)|^2
		grad[i] = (I1wx[i] * I1wx[i]) + (I1wy[i] * I1wy[i]);
		// compute the constant part of the rho function
		rho_c[i] = (I1w[i] - I1wx[i] * u1[i] - I1wy[i] * u2[i] - I0[i]);
	}
}


__global__ void estimateThreshold(const __half2* rho_c, const __half2* I1wx, const __half2* u1, const __half2* I1wy,
	const __half2* u2, const __half2* grad, __half2 lT, size_t size, __half2* v1, __half2* v2)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < size) {
		const __half2 rho = rho_c[i] + (I1wx[i] * u1[i] + I1wy[i] * u2[i]);
		const __half2 fi{-rho/grad[i]};
		const bool c1{rho >= -lT * grad[i]};
		const bool c2{rho > lT * grad[i]};
		const bool c3{grad[i] < __float2half2_rn(GRAD_IS_ZERO)};
		__half2 d1{lT * I1wx[i]}; 
		__half2 d2{lT * I1wy[i]};

		if(c1) {
			d1 = fi * I1wx[i];
			d2 = fi * I1wy[i];

			if(c2) {
				d1 = -lT * I1wx[i];
				d2 = -lT * I1wy[i];
			}
			else if(c3)
				d1 = d2 = __float2half2_rn(0.0f);
		}

		v1[i] = u1[i] + d1;
		v2[i] = u2[i] + d2;
	}
}


__global__ void estimateOpticalFlow(__half2* u1, __half2* u2, const __half2* v1, const __half2* v2, 
	const __half2* div_p1, const __half2* div_p2, __half2 theta, size_t size, float* error)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < size) {		
		const float u1k = __high2float(u1[i]);
		const float u2k = __high2float(u2[i]);

		u1[i] = v1[i] + theta * div_p1[i];
		u2[i] = v2[i] + theta * div_p2[i];

		const float u1n = __high2float(u1[i]);
		const float u2n = __high2float(u2[i]);

		error[i] = (u1n - u1k) * (u1n - u1k) + (u2n - u2k) * (u2n - u2k);
	}
}


__global__ void estimateGArgs(const __half2* div_p1, const __half2* div_p2, const __half2* v1, const __half2* v2, 
	size_t size, __half2 taut, __half2* g1, __half2* g2)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < size){		
		g1[i] = __float2half2_rn(1.0f) + taut * __float2half2_rn(hypotf(__high2float(div_p1[i]), __high2float(v1[i])));
		g2[i] = __float2half2_rn(1.0f) + taut * __float2half2_rn(hypotf(__high2float(div_p2[i]), __high2float(v2[i])));
	}
}


__global__ void divideByG(const __half2* g1, const __half2* g2, size_t size, __half2* p11, __half2* p12, 
	__half2* p21, __half2* p22)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < size){		
		p11[i] = p11[i] / g1[i];
		p12[i] = p12[i] / g1[i];
		p21[i] = p21[i] / g2[i];
		p22[i] = p22[i] / g2[i];
	}
}


__global__ void normKernel(const __half2* __restrict__ I0, const __half2* __restrict__ I1, __half2* __restrict__ I0n, __half2* __restrict__ I1n, __half2 min, __half2 den, int size) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(i < size) {
		I0n[i] = __float2half2_rn(255.0) * (I0[i] - min) / den;
		I1n[i] = __float2half2_rn(255.0) * (I1[i] - min) / den;
	}
}


__global__ void copyFloat2Half2(const float* __restrict__ in, __half2* out, int size) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(i < size) {
		out[i] = __float2half2_rn(in[i]);
	}
}


__device__ __half2 warpMax(__half2 max) {
	max = __hmax2(max, __shfl_down_sync(MASK, max, 16));
	max = __hmax2(max, __shfl_down_sync(MASK, max, 8));
	max = __hmax2(max, __shfl_down_sync(MASK, max, 4));
	max = __hmax2(max, __shfl_down_sync(MASK, max, 2));
	max = __hmax2(max, __shfl_down_sync(MASK, max, 1));
    return max;
}


__device__ __half2 warpMin(__half2 min) {
	min = __hmin2(min, __shfl_down_sync(MASK, min, 16));
	min = __hmin2(min, __shfl_down_sync(MASK, min, 8));
	min = __hmin2(min, __shfl_down_sync(MASK, min, 4));
	min = __hmin2(min, __shfl_down_sync(MASK, min, 2));
	min = __hmin2(min, __shfl_down_sync(MASK, min, 1));
    return min;
}


__device__ bool lastBlock(int* counter) {
    __threadfence(); //ensure that partial result is visible by all BLOCKS
    int last = 0;
    if (threadIdx.x == 0)
        last = atomicAdd(counter, 1);
    return __syncthreads_or(last == gridDim.x-1);
}


__global__ void half2MaxMin(int N, __half2* __restrict__ inVec, __half2* __restrict__ partialMax, __half2* __restrict__ partialMin, int* __restrict__ lastBlockCounter) {
    int thIdx = threadIdx.x;
    const int globalIdx = thIdx + blockIdx.x * blockDim.x;
    const int gridSize = blockDim.x * gridDim.x;
    const int wrapIdx = thIdx / WRAPS_PER_BLOCK;

	if(globalIdx < N) {
		//perform private sums
		__half2 max{__float2half2_rn(-1.0)}, min{__float2half2_rn(1000.0)};
		for (int i = globalIdx; i < N; i += gridSize) {
			max = __hmax2(max, inVec[i]);
			min = __hmin2(min, inVec[i]);
		}

		// share among block threads private sum
		__shared__ __half2 shMax[WRAPS_PER_BLOCK];
		__shared__ __half2 shMin[WRAPS_PER_BLOCK];

		// SIMT reduction
		shMax[wrapIdx] = warpMax(max);
		shMin[wrapIdx] = warpMin(min);
		__syncthreads();

		//first warp only
		if (thIdx < WRAP_SIZE) {
			max = thIdx * WRAP_SIZE < blockDim.x ? shMax[thIdx] : __float2half2_rn(0.0f);
			min = thIdx * WRAP_SIZE < blockDim.x ? shMin[thIdx] : __float2half2_rn(0.0f);
			//join the other wrap reductions
			max = warpMax(max);
			min = warpMin(min);
			
			// each block shares its partial reduction
			if (thIdx == 0) {
				partialMax[blockIdx.x] = max;
				partialMin[blockIdx.x] = min;
			}
		}

		// choose last block to perform the final reduction
		if (lastBlock(lastBlockCounter)) {
			max = thIdx < gridSize ? partialMax[thIdx] : __float2half2_rn(0.0f);
			min = thIdx < gridSize ? partialMin[thIdx] : __float2half2_rn(0.0f);
			shMax[wrapIdx] = warpMax(max);
			shMin[wrapIdx] = warpMin(min);
			__syncthreads();

			//first warp only
			if (thIdx < WRAP_SIZE) {
				max = thIdx * WRAP_SIZE < blockDim.x ? shMax[thIdx] : __float2half2_rn(0.0f);
				min = thIdx * WRAP_SIZE < blockDim.x ? shMin[thIdx] : __float2half2_rn(0.0f);
				//join the other wrap reductions
				max = warpMax(max);
				min = warpMin(min);
				
				if (thIdx == 0) {
					partialMax[0] = max;
					partialMin[0] = min;
				}
			}
		}
	}
}