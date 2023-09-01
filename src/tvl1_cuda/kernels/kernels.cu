#include <iostream>
#include <cmath>

#include "kernels.cuh"


__global__ void bodyDivergence(const float* v1, const float* v2, float* div, int nx, int ny){
	const int i = (blockIdx.x * blockDim.x + threadIdx.x) + 1;
	if(i < (nx-1)*(ny-1)){
		div[i]  = (v1[i] - v1[i-1]) + (v2[i] - v2[i-nx]);
	}
}


__global__ void edgeRowsDivergence(const float* v1, const float* v2, float* div, int nx, int ny){
	const int j = (blockIdx.x * blockDim.x + threadIdx.x) + 1;
	const int p = (ny-1) * nx + j;

	if(j < (nx-1)){
		div[j] = v1[j] - v1[j-1] + v2[j];
		div[p] = v1[p] - v1[p-1] - v2[p-nx];
	}
}


__global__ void edgeColumnsDivergence(const float* v1, const float* v2, float* div, int nx, int ny){
	const int i = (blockIdx.x * blockDim.x + threadIdx.x) + 1;
	const int p1 = i * nx;
	const int p2 = (i+1) * nx - 1;

	if(i < (ny-1)){
		div[p1] =  v1[p1]   + v2[p1] - v2[p1 - nx];
		div[p2] = -v1[p2-1] + v2[p2] - v2[p2 - nx];
	}
}


__global__ void cornersDivergence(const float* v1, const float* v2, float* div, int nx, int ny){
	div[0]         =  v1[0] + v2[0];
	div[nx-1]      = -v1[nx - 2] + v2[nx - 1];
	div[(ny-1)*nx] =  v1[(ny-1)*nx] - v2[(ny-2)*nx];
	div[ny*nx-1]   = -v1[ny*nx - 2] - v2[(ny-1)*nx - 1];
}


__global__ void bodyForwardGradient(const float* f, float* fx, float* fy, size_t nx, size_t ny){
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < (nx-1)*(ny-1)){
		fx[i] = f[i+1] - f[i];
		fy[i] = f[i+nx] - f[i];
	}
}


__global__ void rowsForwardGradient(const float* f, float* fx, float* fy, size_t nx, size_t ny){
	const int j = blockIdx.x * blockDim.x + threadIdx.x;
	const int p = (ny-1) * nx + j;

	if(j < (nx-1)){
		fx[p] = f[p+1] - f[p];
		fy[p] = 0;
	}
}


__global__ void columnsForwardGradient(const float* f, float* fx, float* fy, size_t nx, size_t ny){
	const int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
	const int p = i * nx-1;

	if(i < ny){
		fx[p] = 0;
		fy[p] = f[p+nx] - f[p];
	}
}


__global__ void bodyGradient(const float* input, float* dx, float* dy, int nx, int ny){
	const int i = (blockIdx.x * blockDim.x + threadIdx.x) + 1;
	if(i < (nx-1)*(ny-1)){
		dx[i] = 0.5*(input[i+1] - input[i-1]);
		dy[i] = 0.5*(input[i+nx] - input[i-nx]);
	}
}


__global__ void edgeRowsGradient(const float* input, float* dx, float* dy, int nx, int ny){
	const int j = (blockIdx.x * blockDim.x + threadIdx.x) + 1;
	const int k = (ny - 1) * nx + j;
	if(j < nx-1) {
		dx[j] = 0.5*(input[j+1] - input[j-1]);
		dy[j] = 0.5*(input[j+nx] - input[j]);
		dx[k] = 0.5*(input[k+1] - input[k-1]);
		dy[k] = 0.5*(input[k] - input[k-nx]);
	}
}


__global__ void edgeColumnsGradient(const float* input, float* dx, float* dy, int nx, int ny){
	const int i = (blockIdx.x * blockDim.x + threadIdx.x) + 1;
	const int p = i * nx;
	const int k = (i+1) * nx - 1;
	if(i < ny-1) {
		dx[p] = 0.5*(input[p+1] - input[p]);
		dy[p] = 0.5*(input[p+nx] - input[p-nx]);
		dx[k] = 0.5*(input[k] - input[k-1]);
		dy[k] = 0.5*(input[k+nx] - input[k-nx]);
	}
}


__global__ void cornersGradient(const float* input, float* dx, float* dy, int nx, int ny){
	dx[0] = 0.5*(input[1] - input[0]);
	dy[0] = 0.5*(input[nx] - input[0]);

	dx[nx-1] = 0.5*(input[nx-1] - input[nx-2]);
	dy[nx-1] = 0.5*(input[2*nx-1] - input[nx-1]);

	dx[(ny-1)*nx] = 0.5*(input[(ny-1)*nx + 1] - input[(ny-1)*nx]);
	dy[(ny-1)*nx] = 0.5*(input[(ny-1)*nx] - input[(ny-2)*nx]);

	dx[ny*nx-1] = 0.5*(input[ny*nx-1] - input[ny*nx-1-1]);
	dy[ny*nx-1] = 0.5*(input[ny*nx-1] - input[(ny-1)*nx-1]);
}


__global__ void convolution1D(float* B, int size, float sPi, float den) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < size)
		B[i] = 1 / sPi * expf(-i * i / den);
}


__global__ void lineConvolution(float *I, const float *B, const int* xDim, const int* yDim, int size, float* buffer) {
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

        for (i = size; i < bdx; i++) {
            float sum = B[0] * buffer[i];
            for (j = 1; j < size; j++)
                sum += B[j] * (buffer[i - j] + buffer[i + j]);
            I[k * xdim + i - size] = sum;
        }
    }
}


__global__ void columnConvolution(float* I, const float* B, const int* xDim, const int* yDim, int size, float* buffer) {
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

        for (i = size; i < bdy; i++) {
            float sum = B[0] * buffer[i];
            for (j = 1; j < size; j++)
                sum += B[j] * (buffer[i - j] + buffer[i + j]);
            I[(i - size) * xdim + k] = sum;
        }
    }
}


__global__ void bicubicResample(const float* Is, float *Iout, const int* nxx, const int* nyy, 
	const int* nx, const int* ny, float factor)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < *nyy * *nxx) {
		const int i = idx / *nxx;
		const int j = idx % *nxx;
		const float ii = (float)i / factor;
		const float jj = (float)j / factor;
        Iout[idx] = bicubicInterpolationAt(Is, jj, ii, *nx, *ny, false);
    }
}


__global__ void bicubicResample2(const float* Is, float *Iout, const int* nxx, const int* nyy, 
	const int* nx, const int* ny)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < *nyy * *nxx) {
		const int i = idx / *nxx;
		const int j = idx % *nxx;
		const float ii = (float)i / ((float)*nyy / *ny);
		const float jj = (float)j / ((float)*nxx / *nx);
        Iout[idx] = bicubicInterpolationAt(Is, jj, ii, *nx, *ny, false);
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
	float factor // zoom factor between 0 and 1
)
{
	//compute the new size corresponding to factor
	//we add 0.5 for rounding off to the closest number
	*nxx = (int)(*nx * factor + 0.5);
	*nyy = (int)(*ny * factor + 0.5);
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
 * Bicubic interpolation in two dimensions
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
  * Compute the bicubic interpolation of a point in an image.
  * Detect if the point goes outside the image domain.
**/
__device__ float bicubicInterpolationAt(
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
	float v[4]{0};
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
  * Compute the bicubic interpolation of an image.
**/
__global__ void bicubicInterpolationWarp(
	const float* input,     // image to be warped
	const float *u,         // x component of the vector field
	const float *v,         // y component of the vector field
	float       *output,    // image warped with bicubic interpolation
	int    nx,        // image width
	int    ny,        // image height
	bool border_out // if true, put zeros outside the region
)
{
	const int p = blockIdx.x * blockDim.x + threadIdx.x;

	// obtain the bicubic interpolation at position (uu, vv)
	if(p < nx*ny){
		const float uu = (static_cast<int>(p%nx)) + u[p];
		const float vv = (static_cast<int>(p/nx)) + v[p];
		output[p] = bicubicInterpolationAt(input, uu, vv, nx, ny, border_out);
	}
}


__global__ void calculateRhoGrad(const float* I1wx, const float* I1wy, const float* I1w,
	const float* u1, const float* u2, const float* I0, float* grad, float* rho_c, int size)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < size) {
		// store the |Grad(I1)|^2
		grad[i] = (I1wx[i] * I1wx[i]) + (I1wy[i] * I1wy[i]);
		// compute the constant part of the rho function
		rho_c[i] = (I1w[i] - I1wx[i] * u1[i] - I1wy[i] * u2[i] - I0[i]);
	}
}


__global__ void estimateThreshold(const float* rho_c, const float* I1wx, const float* u1, const float* I1wy,
	const float* u2, const float* grad, float lT, size_t size, float* v1, float* v2)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < size) {
		const float rho = rho_c[i] + (I1wx[i] * u1[i] + I1wy[i] * u2[i]);
		const float fi{-rho/grad[i]};
		const bool c1{rho >= -lT * grad[i]};
		const bool c2{rho > lT * grad[i]};
		const bool c3{grad[i] < GRAD_IS_ZERO};
		float d1{lT * I1wx[i]}; 
		float d2{lT * I1wy[i]};

		if(c1) {
			d1 = fi * I1wx[i];
			d2 = fi * I1wy[i];

			if(c2) {
				d1 = -lT * I1wx[i];
				d2 = -lT * I1wy[i];
			}
			else if(c3)
				d1 = d2 = 0.0f;
		}

		v1[i] = u1[i] + d1;
		v2[i] = u2[i] + d2;
	}
}


__global__ void estimateOpticalFlow(float* u1, float* u2, const float* v1, const float* v2, 
	const float* div_p1, const float* div_p2, float theta, size_t size, float* error)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < size) {		
		const float u1k = u1[i];
		const float u2k = u2[i];

		u1[i] = v1[i] + theta * div_p1[i];
		u2[i] = v2[i] + theta * div_p2[i];

		error[i] = (u1[i] - u1k) * (u1[i] - u1k) + (u2[i] - u2k) * (u2[i] - u2k);
	}
}


__global__ void estimateGArgs(const float* div_p1, const float* div_p2, const float* v1, const float* v2, 
	size_t size, float taut, float* g1, float* g2)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < size){		
		g1[i] = 1.0f + taut * hypotf(div_p1[i], v1[i]);
		g2[i] = 1.0f + taut * hypotf(div_p2[i], v2[i]);
	}
}


__global__ void divideByG(const float* g1, const float* g2, size_t size, float* p11, float* p12, 
	float* p21, float* p22)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < size){		
		p11[i] = p11[i] / g1[i];
		p12[i] = p12[i] / g1[i];
		p21[i] = p21[i] / g2[i];
		p22[i] = p22[i] / g2[i];
	}
}


__global__ void normKernel(const float* __restrict__ I0, const float* __restrict__ I1, float* __restrict__ I0n, float* __restrict__ I1n, float min, float den, int size) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(i < size) {
		I0n[i] = 255.0 * (I0[i] - min) / den;
		I1n[i] = 255.0 * (I1[i] - min) / den;
	}
}
