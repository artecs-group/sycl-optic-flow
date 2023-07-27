# Optical flow algorithms for SYCL
<img alt="license" src="https://img.shields.io/github/license/mashape/apistatus.svg"/>

Optical flow algorithms implemented in SYCL

# Table of contents
1. [Dependencies](#1-dependencies)
    1. [Dependencies for CUDA](#11-dependencies-for-cuda)
2. [Compile and run](#2-compile-and-run)
3. [Acknowledgement](#acknowledgement)

## 1. Dependencies
All the code were tested under Ubuntu 22.04.

* [oneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html). Tested under 2023.2.0 version.
* OpenCV 4.5.4
```bash
sudo apt install libopencv-dev
```
* Make 4.3
```bash
sudo apt install make
```
* CMake 3.22.1
```bash
sudo apt install cmake
```
* Ninja 1.10.1
```bash
sudo apt install ninja-build
```

### 1.1 Dependencies for CUDA
In the case you would run on NVIDIA GPUs you also need:

* [CUDA 12.0](https://developer.nvidia.com/cuda-12-0-0-download-archive)
* [The codeplay's oneAPI plugin for NVIDIA GPUs](https://developer.codeplay.com/products/oneapi/nvidia/home/).

## 2. Compile and run
First of all, load the oneAPI variables:

```bash
source /path/to/oneapi/setvars.sh
```

You can show the available devices by using:

```bash
sycl-ls
> [opencl:cpu:0] Intel(R) OpenCL, Intel(R) Core(TM) i7-10700 CPU @ 2.90GHz 3.0 [2023.16.6.0.22_223734]
  [opencl:gpu:1] Intel(R) OpenCL HD Graphics, Intel(R) UHD Graphics 630 3.0 [23.05.25593.11]
  [opencl:acc:2] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device 1.2 [2023.15.3.0.20_160000]
```
Finally, the following commands show how to build and launch the project. To control in which device we are running, use the [ONEAPI_DEVICE_SELECTOR](https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md#oneapi-device-selector) variable. 

```bash
mkdir build
cd build
cmake -G Ninja -DCMAKE_CXX_COMPILER=icpx ..
cmake --build .
ONEAPI_DEVICE_SELECTOR=opencl:cpu ./tvl1 --camera=0
```

In the case you want to run in a NVIDIA GPU, you must configure the project for it:

```bash
cmake -G Ninja -DCMAKE_CXX_COMPILER=icpx -DDEVICE=ngpu ..
cmake --build .
ONEAPI_DEVICE_SELECTOR=cuda:gpu ./tvl1 --camera=0
```

## Acknowledgement
This work has been supported by the EU (FEDER), the Spanish MINECO and CM under grants S2018/TCS-4423, PID2021-126576NB-I00 funded by MCIN/AEI/10.13039/501100011033 and by "ERDF A way of making Europe".