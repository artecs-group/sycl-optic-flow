cmake_minimum_required(VERSION 3.22.1)
project(TVL1 LANGUAGES CXX)

set(CXX_STANDARD_REQUIRED 17)

set(This lk)

set(Headers
    lucaskanade.hpp
    kernels/kernels.hpp
)

set(Source
    main.cpp
    lucaskanade.cpp
    kernels/kernels.cpp
)

set(EXTRA_FLAGS "")

if(DEBUG)
    message(STATUS "Added -g to compilation.")
    set(EXTRA_FLAGS "-g")
endif()

find_package(OpenCV REQUIRED)
set(CXXFLAGS "-O3" "-fsycl" ${EXTRA_FLAGS})

if(NGPU)
    #list(APPEND CXXFLAGS "-fsycl-targets=nvptx64-nvidia-cuda,nvidia_gpu_sm_53,nvidia_gpu_sm_61,nvidia_gpu_sm_87")
    list(APPEND CXXFLAGS "-fsycl-targets=nvidia_gpu_sm_61")
else()
    find_package(IntelSYCL REQUIRED)
endif()

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(${This} ${Source} ${Headers})
target_compile_options(${This} PRIVATE ${CXXFLAGS})

target_link_directories(${This} PRIVATE)
target_link_libraries(${This} ${OpenCV_LIBS} ${CXXFLAGS} "-fsycl-device-code-split=per_kernel" sycl)
