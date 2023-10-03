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
/// \brief add two vectors of size _count_
///
/// CUDA kernel
/// \param[in]  op1   term one
/// \param[in]  op2   term two
/// \param[in]  count vector size
/// \param[out] sum   result
///////////////////////////////////////////////////////////////////////////////

void AddKernel(const float *op1, const float *op2, int count, float *sum,
               const sycl::nd_item<3> &item_ct1)
{
    const int pos = item_ct1.get_local_id(2) +
                    item_ct1.get_group(2) * item_ct1.get_local_range(2);

    if (pos >= count) return;

    sum[pos] = op1[pos] + op2[pos];
}

///////////////////////////////////////////////////////////////////////////////
/// \brief add two vectors of size _count_
/// \param[in]  op1   term one
/// \param[in]  op2   term two
/// \param[in]  count vector size
/// \param[out] sum   result
///////////////////////////////////////////////////////////////////////////////
static
void Add(sycl::queue q, const float *op1, const float *op2, int count, float *sum)
{
    sycl::range<3> threads(1, 1, 256);
    sycl::range<3> blocks(1, 1, iDivUp(count, threads[2]));

    /*
    DPCT1049:6: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    q.parallel_for(
        sycl::nd_range<3>(blocks * threads, threads),
        [=](sycl::nd_item<3> item_ct1) {
            AddKernel(op1, op2, count, sum, item_ct1);
        });
}
