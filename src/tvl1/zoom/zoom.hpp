#ifndef ZOOM_C
#define ZOOM_C

#define ZOOM_SIGMA_ZERO 0.6

void zoom_out(const float *I, float *Iout, const int nx, const int ny, const float factor, float* Is, float* gaussBuffer);
void zoom_in(const float *I, float *Iout, int nx, int ny, int nxx, int nyy);

/**
  *
  * Compute the size of a zoomed image from the zoom factor
  *
**/
inline void zoom_size(
	int nx,      // width of the orignal image
	int ny,      // height of the orignal image
	int *nxx,    // width of the zoomed image
	int *nyy,    // height of the zoomed image
	float factor // zoom factor between 0 and 1
)
{
	//compute the new size corresponding to factor
	//we add 0.5 for rounding off to the closest number
	*nxx = (int)((float) nx * factor + 0.5);
	*nyy = (int)((float) ny * factor + 0.5);
}

#endif