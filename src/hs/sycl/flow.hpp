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

#ifndef FLOW_H
#define FLOW_H

#include <sycl/sycl.hpp>

void initFlow(sycl::queue q, int nLevels, int stride, int width, int height);

void ComputeFlow(sycl::queue q,
                    const float *I0,  // source frame
                     const float *I1,  // tracked frame
                     int width,        // frame width
                     int height,       // frame height
                     int stride,       // row access stride
                     float alpha,      // smoothness coefficient
                     int nLevels,      // number of levels in pyramid
                     int nWarpIters,   // number of warping iterations per pyramid level
                     int nSolverIters, // number of solver iterations (for linear system)
                     float *u,         // output horizontal flow
                     float *v);        // output vertical flow

void deleteFlow_mem(sycl::queue q, int nLevels);
#endif
