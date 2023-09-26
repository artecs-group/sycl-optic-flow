/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
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
void AddKernel(const float *op1, const float *op2, int count,
                          float *sum, const sycl::nd_item<3> &item_ct1) {
  const int pos = item_ct1.get_local_id(2) +
                  item_ct1.get_group(2) * item_ct1.get_local_range(2);

  if (pos < count)
    sum[pos] = op1[pos] + op2[pos];
}

///////////////////////////////////////////////////////////////////////////////
/// \brief add two vectors of size _count_
/// \param[in]  op1   term one
/// \param[in]  op2   term two
/// \param[in]  count vector size
/// \param[out] sum   result
///////////////////////////////////////////////////////////////////////////////
static void Add(const float *op1, const float *op2, int count, float *sum, sycl::queue q) {
  sycl::range<3> threads(1, 1, 256);
  sycl::range<3> blocks(1, 1, iDivUp(count, threads[2]));

  q.parallel_for(
      sycl::nd_range<3>(blocks * threads, threads),
      [=](sycl::nd_item<3> item_ct1) {
        AddKernel(op1, op2, count, sum, item_ct1);
      });
}
