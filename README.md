# Optical flow algorithms for SYCL
<img alt="license" src="https://img.shields.io/github/license/mashape/apistatus.svg"/>

Optical flow algorithms implemented in different languages.

# Table of contents
1. [Dependencies](#1-dependencies)
    1. [CUDA dependencies](#11-cuda-dependencies)
2. [Acknowledgement](#acknowledgement)

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

### 1.1 CUDA dependencies
In the case you would run on NVIDIA GPUs you also need:

* [CUDA 12.0](https://developer.nvidia.com/cuda-12-0-0-download-archive)
* To run SYCL over NVIDIA GPU you must install [the standalone compiler](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md) and the [oneMKL library](https://github.com/oneapi-src/oneMKL).

## Acknowledgement
This work has been supported by the EU (FEDER), the Spanish MINECO and CM under grants S2018/TCS-4423, PID2021-126576NB-I00 funded by MCIN/AEI/10.13039/501100011033 and by "ERDF A way of making Europe".