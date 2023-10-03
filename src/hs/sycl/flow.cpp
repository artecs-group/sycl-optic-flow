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

#include <sycl/sycl.hpp>
#include "common.hpp"

// include kernels
#include "kernels/downscaleKernel.hpp"
#include "kernels/upscaleKernel.hpp"
#include "kernels/warpingKernel.hpp"
#include "kernels/derivativesKernel.hpp"
#include "kernels/solverKernel.hpp"
#include "kernels/addKernel.hpp"

int *pW, *pH, *pS;

// device memory pointers
float *pI0, *pI1;

float *d_tmp;
float *d_du0;
float *d_dv0;
float *d_du1;
float *d_dv1;

float *d_Ix;
float *d_Iy;
float *d_Iz;

float *d_u;
float *d_v;
float *d_nu;
float *d_nv;

void initFlow(sycl::queue q, int nLevels, int stride, int width, int height)
{
    const int dataSize = stride * height * sizeof(float);

    pW = new int [nLevels];
    pH = new int [nLevels];
    pS = new int [nLevels];

    // allocate GPU memory for input images
    pI0 = (float *)sycl::malloc_device(nLevels * dataSize, q);
    pI1 = (float *)sycl::malloc_device(nLevels * dataSize, q);
    d_tmp = (float *)sycl::malloc_device(dataSize, q);
    d_du0 = (float *)sycl::malloc_device(dataSize, q);
    d_dv0 = (float *)sycl::malloc_device(dataSize, q);
    d_du1 = (float *)sycl::malloc_device(dataSize, q);
    d_dv1 = (float *)sycl::malloc_device(dataSize, q);
    d_Ix = (float *)sycl::malloc_device(dataSize, q);
    d_Iy = (float *)sycl::malloc_device(dataSize, q);
    d_Iz = (float *)sycl::malloc_device(dataSize, q);
    d_u = (float *)sycl::malloc_device(dataSize, q);
    d_v = (float *)sycl::malloc_device(dataSize, q);
    d_nu = (float *)sycl::malloc_device(dataSize, q);
    d_nv = (float *)sycl::malloc_device(dataSize, q);
}

void deleteFlow_mem(sycl::queue q, int nLevels)
{
    delete [] pW;
    delete [] pH;
    delete [] pS;

    sycl::free(pI0, q);
    sycl::free(pI1, q);
    sycl::free(d_tmp, q);
    sycl::free(d_du0, q);
    sycl::free(d_dv0, q);
    sycl::free(d_du1, q);
    sycl::free(d_dv1, q);
    sycl::free(d_Ix, q);
    sycl::free(d_Iy, q);
    sycl::free(d_Iz, q);
    sycl::free(d_nu, q);
    sycl::free(d_nv, q);
    sycl::free(d_u, q);
    sycl::free(d_v, q);
}

///////////////////////////////////////////////////////////////////////////////
/// \brief method logic
///
/// handles memory allocations, control flow
/// \param[in]  I0           source image
/// \param[in]  I1           tracked image
/// \param[in]  width        images width
/// \param[in]  height       images height
/// \param[in]  stride       images stride
/// \param[in]  alpha        degree of displacement field smoothness
/// \param[in]  nLevels      number of levels in a pyramid
/// \param[in]  nWarpIters   number of warping iterations per pyramid level
/// \param[in]  nSolverIters number of solver iterations (Jacobi iterations)
/// \param[out] u            horizontal displacement
/// \param[out] v            vertical displacement
///////////////////////////////////////////////////////////////////////////////
void ComputeFlow(sycl::queue q, const float *I0,
                     const float *I1,
                     int width, int height, int stride,
                     float alpha,
                     int nLevels,
                     int nWarpIters,
                     int nSolverIters,
                     float *u,
                     float *v)
{
    const int dataSize = stride * height * sizeof(float);
    const size_t size = stride * height;
    // prepare pyramid
    int currentLevel = nLevels - 1;

    q.memcpy(pI0 + currentLevel * size, I0, dataSize);
    q.memcpy(pI1 + currentLevel * size, I1, dataSize);

    pW[currentLevel] = width;
    pH[currentLevel] = height;
    pS[currentLevel] = stride;

    for (; currentLevel > 0; --currentLevel)
    {
        int nw = pW[currentLevel] / 2;
        int nh = pH[currentLevel] / 2;
        int ns = iAlignUp(nw);

        Downscale(q, pI0 + currentLevel*size, pW[currentLevel], pH[currentLevel],
                  pS[currentLevel], nw, nh, ns, pI0 + (currentLevel - 1)*size);

        Downscale(q, pI1 + currentLevel*size, pW[currentLevel], pH[currentLevel],
                  pS[currentLevel], nw, nh, ns, pI1 + (currentLevel - 1)*size);

        pW[currentLevel - 1] = nw;
        pH[currentLevel - 1] = nh;
        pS[currentLevel - 1] = ns;
    }

    q.memset(d_u, 0, stride * height * sizeof(float));
    q.memset(d_v, 0, stride * height * sizeof(float));

    // compute flow
    for (; currentLevel < nLevels; ++currentLevel)
    {

        for (int warpIter = 0; warpIter < nWarpIters; ++warpIter)
        {
            q.memset(d_du0, 0, dataSize);
            q.memset(d_dv0, 0, dataSize);
            q.memset(d_du1, 0, dataSize);
            q.memset(d_dv1, 0, dataSize);

            // on current level we compute optical flow
            // between frame 0 and warped frame 1
            WarpImage(q, pI1 + currentLevel*size, pW[currentLevel], pH[currentLevel],
                      pS[currentLevel], d_u, d_v, d_tmp);

            ComputeDerivatives(q, pI0 + currentLevel*size, d_tmp, pW[currentLevel],
                               pH[currentLevel], pS[currentLevel], d_Ix, d_Iy, d_Iz);

            for (int iter = 0; iter < nSolverIters; ++iter)
            {
                SolveForUpdate(q, d_du0, d_dv0, d_Ix, d_Iy, d_Iz,
                               pW[currentLevel], pH[currentLevel], pS[currentLevel], alpha, d_du1, d_dv1);

                Swap(d_du0, d_du1);
                Swap(d_dv0, d_dv1);
            }

            // update u, v
            Add(q, d_u, d_du0, pH[currentLevel] * pS[currentLevel], d_u);
            Add(q, d_v, d_dv0, pH[currentLevel] * pS[currentLevel], d_v);
        }

        if (currentLevel != nLevels - 1)
        {
            // prolongate solution
            float scaleX = (float)pW[currentLevel + 1]/(float)pW[currentLevel];

            Upscale(q, d_u, pW[currentLevel], pH[currentLevel], pS[currentLevel],
                    pW[currentLevel + 1], pH[currentLevel + 1], pS[currentLevel + 1], scaleX, d_nu);

            float scaleY = (float)pH[currentLevel + 1]/(float)pH[currentLevel];

            Upscale(q, d_v, pW[currentLevel], pH[currentLevel], pS[currentLevel],
                    pW[currentLevel + 1], pH[currentLevel + 1], pS[currentLevel + 1], scaleY, d_nv);

            Swap(d_u, d_nu);
            Swap(d_v, d_nv);
        }
    }

    q.memcpy(u, d_u, dataSize).wait();
    q.memcpy(v, d_v, dataSize).wait();
}
