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
/// \brief downscale image
///
/// CUDA kernel, relies heavily on texture unit
/// \param[in]  width   image width
/// \param[in]  height  image height
/// \param[in]  stride  image stride
/// \param[out] out     result
///////////////////////////////////////////////////////////////////////////////
void DownscaleKernel(int width, int height, int stride, float *out, const float* src,
                     const sycl::nd_item<3> &item_ct1)
{
    const int ix = item_ct1.get_local_id(2) +
                   item_ct1.get_group(2) * item_ct1.get_local_range(2);
    const int iy = item_ct1.get_local_id(1) +
                   item_ct1.get_group(1) * item_ct1.get_local_range(1);

    if (ix >= width-1 || iy >= height-1)
        return;

    const size_t srcx = ix * 2;
    const size_t srcy = iy * 2;

    out[ix + iy * stride] = 0.25f * (src[srcx + srcy*stride] + src[srcx + (srcy+1)*stride] +
                                    src[(srcx+1) + srcy*stride] + src[(srcx+1) + (srcy+1)*stride]);
}

///////////////////////////////////////////////////////////////////////////////
/// \brief downscale image
///
/// \param[in]  src     image to downscale
/// \param[in]  width   image width
/// \param[in]  height  image height
/// \param[in]  stride  image stride
/// \param[out] out     result
///////////////////////////////////////////////////////////////////////////////
static
void Downscale(sycl::queue q, const float *src, int width, int height, int stride,
               int newWidth, int newHeight, int newStride, float *out)
{
    sycl::range<3> threads(1, 8, 32);
    sycl::range<3> blocks(1, iDivUp(newHeight, threads[1]),
                          iDivUp(newWidth, threads[2]));

    /*
    DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    q.parallel_for(
        sycl::nd_range<3>(blocks * threads, threads),
        [=](sycl::nd_item<3> item_ct1) {
            DownscaleKernel(newWidth, newHeight, newStride, out, src, item_ct1);
        });
}
