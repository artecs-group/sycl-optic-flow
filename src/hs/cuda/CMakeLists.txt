cmake_minimum_required(VERSION 3.22.1)
project(Horn-Schnuck LANGUAGES CXX CUDA)
enable_language(CUDA)

set(CXX_STANDARD_REQUIRED 17)

# Set the CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 53 61 87)

set(This hs)

set(Headers
    helper_cuda.cuh
    helper_string.cuh
    flow.cuh
    common.cuh
    kernels/addKernel.cuh
    kernels/derivativesKernel.cuh
    kernels/downscaleKernel.cuh
    kernels/solverKernel.cuh
    kernels/upscaleKernel.cuh
    kernels/warpingKernel.cuh
)

set(Source
    main.cu
    flow.cu
)

set(EXTRA_FLAGS "")

if(DEBUG)
    message(STATUS "Added -g to compilation.")
    set(EXTRA_FLAGS "-g")
endif()

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(${This} ${Source} ${Headers})

set(CXXFLAGS "-O3" ${EXTRA_FLAGS})
target_compile_options(${This} PRIVATE ${CXXFLAGS})
target_link_directories(${This} PRIVATE)
target_link_libraries(${This} ${OpenCV_LIBS} ${CXXFLAGS})
