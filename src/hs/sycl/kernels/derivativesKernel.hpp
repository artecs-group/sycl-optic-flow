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
#include "../common.hpp"

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
void ComputeDerivativesKernel(int width, int height, int stride,
                                         float *Ix, float *Iy, float *Iz,
                                         const float* src, 
                                         const float* target,
                                         const sycl::nd_item<3> &item_ct1)
{
    const int ix = item_ct1.get_local_id(2) +
                   item_ct1.get_group(2) * item_ct1.get_local_range(2);
    const int iy = item_ct1.get_local_id(1) +
                   item_ct1.get_group(1) * item_ct1.get_local_range(1);

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
void ComputeDerivatives(sycl::queue q, const float *I0, const float *I1,
                        int w, int h, int s,
                        float *Ix, float *Iy, float *Iz)
{
    sycl::range<3> threads(1, 6, 32);
    sycl::range<3> blocks(1, iDivUp(h, threads[1]), iDivUp(w, threads[2]));

    /*
    DPCT1049:3: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    q.parallel_for(
        sycl::nd_range<3>(blocks * threads, threads),
        [=](sycl::nd_item<3> item_ct1) {
            ComputeDerivativesKernel(w, h, s, Ix, Iy, Iz, I0, I1, item_ct1);
        });
}
