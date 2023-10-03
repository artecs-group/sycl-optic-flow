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
/// \brief upscale one component of a displacement field, CUDA kernel
/// \param[in]  width   field width
/// \param[in]  height  field height
/// \param[in]  stride  field stride
/// \param[in]  scale   scale factor (multiplier)
/// \param[out] out     result
///////////////////////////////////////////////////////////////////////////////
void UpscaleKernel(int width, int height, int stride, float scale, float *out, const float* src,
                   const sycl::nd_item<3> &item_ct1)
{
    const int ix = item_ct1.get_local_id(2) +
                   item_ct1.get_group(2) * item_ct1.get_local_range(2);
    const int iy = item_ct1.get_local_id(1) +
                   item_ct1.get_group(1) * item_ct1.get_local_range(1);

    if (ix >= width || iy >= height) return;

    float x = ((float)ix - 0.5f) * 0.5f;
    float y = ((float)iy - 0.5f) * 0.5f;
    const size_t index = x + y*stride;

    // exploit hardware interpolation
    // and scale interpolated vector to match next pyramid level resolution
    out[ix + iy * stride] = src[index] * scale;
}

///////////////////////////////////////////////////////////////////////////////
/// \brief upscale one component of a displacement field, kernel wrapper
/// \param[in]  src         field component to upscale
/// \param[in]  width       field current width
/// \param[in]  height      field current height
/// \param[in]  stride      field current stride
/// \param[in]  newWidth    field new width
/// \param[in]  newHeight   field new height
/// \param[in]  newStride   field new stride
/// \param[in]  scale       value scale factor (multiplier)
/// \param[out] out         upscaled field component
///////////////////////////////////////////////////////////////////////////////
static
void Upscale(sycl::queue q, const float *src, int width, int height, int stride,
             int newWidth, int newHeight, int newStride, float scale, float *out)
{
    sycl::range<3> threads(1, 8, 32);
    sycl::range<3> blocks(1, iDivUp(newHeight, threads[1]),
                          iDivUp(newWidth, threads[2]));

    /*
    DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    q.parallel_for(
        sycl::nd_range<3>(blocks * threads, threads),
        [=](sycl::nd_item<3> item_ct1) {
            UpscaleKernel(newWidth, newHeight, newStride, scale, out, src,
                          item_ct1);
        });
}
