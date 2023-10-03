/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */

#include "../common.cuh"

///////////////////////////////////////////////////////////////////////////////
/// \brief compute image derivatives
///
/// CUDA kernel, relies heavily on texture unit
/// \param[in]  width   image width
/// \param[in]  height  image height
/// \param[in]  stride  image stride
/// \param[out] Ix      x derivative
/// \param[out] Iy      y derivative
/// \param[out] Iz      temporal derivative
///////////////////////////////////////////////////////////////////////////////
__global__ void ComputeDerivativesKernel(int width, int height, int stride,
                                         float *Ix, float *Iy, float *Iz,
                                         const float* src, 
                                         const float* target)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    const int pos = ix + iy * stride;

    if (ix >= width-2 || iy >= height-2) return;

    float t0, t1;
    t0  = src[(ix - 2) + iy*stride ];
    t0 -= src[(ix - 1) + iy*stride ] * 8.0f;
    t0 += src[(ix + 1) + iy*stride ] * 8.0f; 
    t0 -= src[(ix + 2) + iy*stride ];
    t0 /= 12.0f;

    t1  = target[(ix - 2) + iy*stride ];
    t1 -= target[(ix - 1) + iy*stride ] * 8.0f;
    t1 += target[(ix + 1) + iy*stride ] * 8.0f; 
    t1 -= target[(ix + 2) + iy*stride ];
    t1 /= 12.0f;

    Ix[pos] = (t0 + t1) * 0.5f;

    // t derivative
    Iz[pos] = target[ix + iy*stride ] - src[ix + iy*stride ];

    // y derivative
    t0  = src[ix + (iy - 2)*stride ];
    t0 -= src[ix + (iy - 1)*stride ] * 8.0f;
    t0 += src[ix + (iy + 1)*stride ] * 8.0f; 
    t0 -= src[ix + (iy + 2)*stride ];
    t0 /= 12.0f;

    t1  = target[ix + (iy - 2)*stride ];
    t1 -= target[ix + (iy - 1)*stride ] * 8.0f;
    t1 += target[ix + (iy + 1)*stride ] * 8.0f; 
    t1 -= target[ix + (iy + 2)*stride ];
    t1 /= 12.0f;

    Iy[pos] = (t0 + t1) * 0.5f;
}

///////////////////////////////////////////////////////////////////////////////
/// \brief compute image derivatives
///
/// \param[in]  I0  source image
/// \param[in]  I1  tracked image
/// \param[in]  w   image width
/// \param[in]  h   image height
/// \param[in]  s   image stride
/// \param[out] Ix  x derivative
/// \param[out] Iy  y derivative
/// \param[out] Iz  temporal derivative
///////////////////////////////////////////////////////////////////////////////
static
void ComputeDerivatives(const float *I0, const float *I1,
                        int w, int h, int s,
                        float *Ix, float *Iy, float *Iz)
{
    dim3 threads(32, 6);
    dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

    ComputeDerivativesKernel<<<blocks, threads>>>(w, h, s, Ix, Iy, Iz, I0, I1);
}
