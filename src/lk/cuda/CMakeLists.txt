cmake_minimum_required(VERSION 3.15)
project(lucas-kanade)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/local/cuda/include)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -ccbin g++-10 -std=c++11" )
cuda_add_executable(lk kernels.cu lucaskanade.cu main.cu)
target_link_libraries(lk ${OpenCV_LIBS})

set(CMAKE_VERBOSE_MAKEFILE ON)
