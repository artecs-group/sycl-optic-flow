cmake_minimum_required(VERSION 3.22.1)
project(TVL1 LANGUAGES CXX)

set(CXX_STANDARD_REQUIRED 17)

set(This tvl1)

set(Headers
    tvl1.hpp
    kernels/kernels.hpp
)

set(Source
    main.cpp
    tvl1.cpp
    kernels/kernels.cpp
)

set(EXTRA_FLAGS "")

if(DEBUG)
    message(STATUS "Added -g to compilation.")
    set(EXTRA_FLAGS "-g" "-qopt-report=3")
endif()

set(THREADS_PER_BLOCK_DEFAULT 128 CACHE STRING "Default number of threads per block")
set(THREADS_PER_BLOCK ${THREADS_PER_BLOCK_DEFAULT} CACHE STRING "Number of threads per block")

find_package(OpenCV REQUIRED)
set(CXXFLAGS "-O3" ${EXTRA_FLAGS})
set(LXXFLAGS "")

if(ACPP)
    find_package(OpenSYCL REQUIRED)
    add_compile_definitions(ACPP=True)
    include_directories($ENV{ONEMKL}/include)
else()
    list(APPEND LXXFLAGS "-fsycl-device-code-split=per_kernel")
    if(NGPU)
        #list(APPEND CXXFLAGS "-fsycl-targets=nvptx64-nvidia-cuda,nvidia_gpu_sm_53,nvidia_gpu_sm_61,nvidia_gpu_sm_87")
        list(APPEND CXXFLAGS "-fsycl" "-fsycl-targets=nvidia_gpu_sm_61")
        include_directories($ENV{ONEMKL}/include)
    else()
        find_package(IntelSYCL REQUIRED)
        list(APPEND CXXFLAGS "-qmkl" "-DMKL_ILP64")
        include_directories($ENV{MKLROOT}/include)
    endif()
endif()

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(${This} ${Source} ${Headers})
add_compile_definitions(THREADS_PER_BLOCK=${THREADS_PER_BLOCK})
target_compile_options(${This} PRIVATE ${CXXFLAGS})

if(ACPP)
    target_link_directories(${This} PRIVATE $ENV{ONEMKL}/lib)
    target_link_libraries(${This} ${OpenCV_LIBS} ${CXXFLAGS} onemkl)
else()
    if(NGPU)
        target_link_directories(${This} PRIVATE $ENV{ONEMKL}/lib)
        target_link_libraries(${This} ${OpenCV_LIBS} ${CXXFLAGS} ${LXXFLAGS} onemkl sycl)
    else()
        target_link_directories(${This} PRIVATE $ENV{MKLROOT}/lib/intel64 $ENV{TBBROOT}/lib/intel64/gcc4.8)
        target_link_libraries(${This} ${OpenCV_LIBS} ${CXXFLAGS} ${LXXFLAGS} mkl_sycl "-Wl,-export-dynamic -Wl,--start-group" mkl_intel_ilp64 mkl_tbb_thread mkl_core "-Wl,--end-group" tbb sycl OpenCL pthread m dl)
    endif()
endif()
