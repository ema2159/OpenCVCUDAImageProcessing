cmake_minimum_required(VERSION 2.8)
project(CUDAOpenCVImageProcessing)

# Use lower GCC version for CUDA
set(CMAKE_C_COMPILER "/usr/bin/gcc-8")
set(CMAKE_C++_COMPILER "/usr/bin/g++-8")

# Include OpenCV
find_package(OpenCV REQUIRED PATHS "/home/ema2159/.local/opencv/opencv-4.2.0/build")
include_directories(${OpenCV_INCLUDE_DIRS})
# Include CUDA
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};)
    cuda_compile(image ../convolutionGauss/gaussFilter.cu)
    cuda_add_library(gaussCUDA ../convolutionGauss/gaussFilter.cu)
    add_definitions(-DGPU_OPENCV_ENABLE)
endif()
  
# Link opencv libs to all executables
link_libraries(${OpenCV_LIBS})

# Create executables
add_executable(gaussFilter ../convolutionGauss/gaussFilter.cpp)
target_link_libraries(gaussFilter gaussCUDA)
