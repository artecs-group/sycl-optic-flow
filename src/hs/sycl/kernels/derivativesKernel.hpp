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
void ComputeDerivativesKernel(int width, int height, int stride, float *Ix,
                              float *Iy, float *Iz,
                              float* src, float* target,
                              const sycl::nd_item<3> &item_ct1) {
  const int ix = item_ct1.get_local_id(2) +
                 item_ct1.get_group(2) * item_ct1.get_local_range(2);
  const int iy = item_ct1.get_local_id(1) +
                 item_ct1.get_group(1) * item_ct1.get_local_range(1);

  const int pos = ix + iy * stride;

  if (ix >= width || iy >= height) return;

  float t0, t1;

  t0  = src[(ix - 2)*height + iy ];
  t0 -= src[(ix - 1)*height + iy ] * 8.0f;
  t0 += src[(ix + 1)*height + iy ] * 8.0f; 
  t0 -= src[(ix + 2)*height + iy ];
  t0 /= 12.0f;

  t1  = target[(ix - 2)*height + iy ];
  t1 -= target[(ix - 1)*height + iy ] * 8.0f;
  t1 += target[(ix + 1)*height + iy ] * 8.0f; 
  t1 -= target[(ix + 2)*height + iy ];
  t1 /= 12.0f;

  Ix[pos] = (t0 + t1) * 0.5f;

  // t derivative
  Iz[pos] = target[ix*height + iy ] - src[ix*height + iy ];

  // y derivative
  t0  = src[ix*height + (iy - 2) ];
  t0 -= src[ix*height + (iy - 1) ] * 8.0f;
  t0 += src[ix*height + (iy + 1) ] * 8.0f; 
  t0 -= src[ix*height + (iy + 2) ];
  t0 /= 12.0f;

  t1  = target[ix*height + (iy - 2) ];
  t1 -= target[ix*height + (iy - 1) ] * 8.0f;
  t1 += target[ix*height + (iy + 1) ] * 8.0f; 
  t1 -= target[ix*height + (iy + 2) ];
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
static void ComputeDerivatives(const float *I0, const float *I1, float *I0_h, float *I1_h,
                               float *src_d0, float *src_d1, int w, int h,
                               int s, float *Ix, float *Iy, float *Iz, sycl::queue q) {
  sycl::range<3> threads(1, 6, 32);
  sycl::range<3> blocks(1, iDivUp(h, threads[1]), iDivUp(w, threads[2]));

  int dataSize = s * h * sizeof(float);

  q.memcpy(I0_h, I0, dataSize);
  q.memcpy(I1_h, I1, dataSize);
  q.wait();

  q.memcpy(src_d0, I0_h, h*w*sizeof(float));
  q.memcpy(src_d1, I1_h, h*w*sizeof(float));
  q.wait();

  // for (int i = 0; i < h; i++) {
  //   for (int j = 0; j < w; j++) {
  //     int index = i * s + j;
  //     src_d0[index] = I0_h[index];
  //   }
  // }

  // for (int i = 0; i < h; i++) {
  //   for (int j = 0; j < w; j++) {
  //     int index = i * s + j;
  //     src_d1[index] = I1_h[index];
  //   }
  // }
  
  q.parallel_for(sycl::nd_range<3>(blocks * threads, threads),[=](sycl::nd_item<3> item_ct1) {
    ComputeDerivativesKernel(
        w, h, s, Ix, Iy, Iz,
        src_d0, src_d1, item_ct1);
  });
}