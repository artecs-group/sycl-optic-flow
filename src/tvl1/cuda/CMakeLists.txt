cmake_minimum_required(VERSION 3.22.1)
project(TVL1 LANGUAGES CXX CUDA)
enable_language(CUDA)

set(CXX_STANDARD_REQUIRED 17)

# Set the CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 53 61 87)

set(This tvl1)

set(Headers
    tvl1.cuh
    kernels/kernels.cuh
)

set(Source
    main.cu
    tvl1.cu
    kernels/kernels.cu
)

set(EXTRA_FLAGS "")

if(DEBUG)
    message(STATUS "Added -g to compilation.")
    set(EXTRA_FLAGS "-g")
endif()

set(THREADS_PER_BLOCK_DEFAULT 128 CACHE STRING "Default number of threads per block")
set(THREADS_PER_BLOCK ${THREADS_PER_BLOCK_DEFAULT} CACHE STRING "Number of threads per block")

find_package(CUDAToolkit REQUIRED)
find_library(CUBLAS_LIBRARY cublas)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(${This} ${Source} ${Headers})

set(CXXFLAGS "-O3" "-lcublas" "-fopenmp" ${EXTRA_FLAGS})
add_compile_definitions(THREADS_PER_BLOCK=${THREADS_PER_BLOCK})
target_compile_options(${This} PRIVATE ${CXXFLAGS})
target_link_directories(${This} PRIVATE)
target_link_libraries(${This} ${OpenCV_LIBS} ${CUBLAS_LIBRARY} ${CXXFLAGS})
