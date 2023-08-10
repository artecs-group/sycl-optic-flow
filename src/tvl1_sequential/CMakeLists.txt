cmake_minimum_required(VERSION 3.22.1)
project(TVL1 LANGUAGES CXX)

set(CXX_STANDARD_REQUIRED 17)

set(This tvl1)

set(Headers
    tvl1.hpp
    zoom/zoom.hpp
    mask/mask.hpp
    bicubic_interpolation/bicubic_interpolation.hpp
)

set(Source
    main.cpp
    tvl1.cpp
    zoom/zoom.cpp
    mask/mask.cpp
    bicubic_interpolation/bicubic_interpolation.cpp
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
elseif(DEVICE STREQUAL "cpu")
    message(STATUS "Configuring for CPU architecture.")
    set(EXTRA_FLAGS "-xHost")
endif()

if(DEBUG)
    message(STATUS "Added -g to compilation.")
    set(EXTRA_FLAGS "-g" "-qopt-report=3")
endif()

find_package(IntelSYCL REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} $ENV{MKLROOT}/include)
add_executable(${This} ${Source} ${Headers})
add_sycl_to_target(TARGET ${This} SOURCES "main.cpp")

set(CXXFLAGS "-O3" "-fsycl-unnamed-lambda" "-fsycl" "-fopenmp" "-qmkl" "-DMKL_ILP64" ${EXTRA_FLAGS})
target_compile_options(${This} PRIVATE ${CXXFLAGS})
target_link_directories(${This} PRIVATE $ENV{MKLROOT}/lib/intel64)
target_link_libraries(${This} ${OpenCV_LIBS} ${CXXFLAGS} "-fsycl-device-code-split=per_kernel" "-Wl,--start-group" mkl_intel_lp64 mkl_sequential mkl_core "-Wl,--end-group" pthread m dl sycl)
