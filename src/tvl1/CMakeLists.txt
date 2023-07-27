cmake_minimum_required(VERSION 3.22.1)
project(TVL1 LANGUAGES CXX)

set(CXX_STANDARD_REQUIRED 17)

set(This tvl1)

set(Source
    main.cpp
    # tvl1.cpp
    # zoom.c
    # mask.c
    # bicubic_interpolation.c
)

set(supported_devices
    cpu
    igpu
    ngpu
)

if(DEVICE)
    list(FIND supported_devices ${DEVICE} device_idx)
    if(device_idx EQUAL -1)
        message(FATAL_ERROR "Available devices are: ${supported_devices}")
    endif()
else()
    set(DEVICE cpu)
endif()

set(EXTRA_FLAGS "")

if(DEVICE STREQUAL "ngpu")
    message(STATUS "Configuring for CUDA architecture.")
    find_package(CUDA REQUIRED)
    set(EXTRA_FLAGS "-fsycl-targets=nvptx64-nvidia-cuda")
endif()


find_package(IntelSYCL REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(${This} ${Source})
#add_sycl_to_target(TARGET ${This} SOURCES ${Sources})

set(CXXFLAGS "-O3" "-fsycl-unnamed-lambda" "-fsycl" ${EXTRA_FLAGS})
target_compile_options(${This} PRIVATE ${CXXFLAGS})
target_link_libraries(${This} ${OpenCV_LIBS} ${CXXFLAGS} "-fsycl-device-code-split=per_kernel" sycl)
