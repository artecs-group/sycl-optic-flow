cmake_minimum_required(VERSION 3.22.1)
project(lucas-kanade LANGUAGES CXX CUDA)
enable_language(CUDA)

set(CXX_STANDARD_REQUIRED 17)

# Set the CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 53 61 87)

set(This lk)

set(Headers
    lucaskanade.cuh
    kernels/kernels.cuh
)

set(Source
    main.cu
    lucaskanade.cu
    kernels/kernels.cu
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

set(CXXFLAGS "-O3" "-lcublas" ${EXTRA_FLAGS})
target_compile_options(${This} PRIVATE ${CXXFLAGS})
target_link_directories(${This} PRIVATE)
target_link_libraries(${This} ${OpenCV_LIBS} ${CXXFLAGS})
