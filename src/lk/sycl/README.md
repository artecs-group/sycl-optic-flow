# Compile and run
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
cmake -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DNGPU=ON ..
cmake --build .
ONEAPI_DEVICE_SELECTOR=cuda:gpu ./tvl1 --video=../../../../dataset/schoolgirls.mp4
```

**Note**: Running on NVIDIA GPU remember to install and load the standalone compiler.

In the other side, compiling for AdaptiveCpp (hipSYCL) you must:

```bash
export OPENSYCL_TARGETS=omp.accelerated;cuda:sm_61
cmake -G Ninja -DACPP=ON -DOpenSYCL_DIR=$ACPP/lib/cmake/OpenSYCL/ -DCMAKE_CXX_COMPILER=syclcc -DNGPU=ON ..
cmake --build .
./tvl1 --video=../../../../dataset/schoolgirls.mp4 --show=false
```

## Acknowledgement
This work has been supported by the EU (FEDER), the Spanish MINECO and CM under grants S2018/TCS-4423, PID2021-126576NB-I00 funded by MCIN/AEI/10.13039/501100011033 and by "ERDF A way of making Europe".