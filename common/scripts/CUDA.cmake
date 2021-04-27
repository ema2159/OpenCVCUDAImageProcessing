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
    # Gaussian filter
    cuda_compile(gauss ../convolutionGauss/gaussFilter.cu)
    cuda_add_library(gaussCUDA ../convolutionGauss/gaussFilter.cu)
    # Laplacian filter
    cuda_compile(laplace ../convolutionLaplace/laplaceFilter.cu)
    cuda_add_library(laplaceCUDA ../convolutionLaplace/laplaceFilter.cu)
    # Image scaling
    cuda_compile(imScaling ../imageTransformation/imScaling.cu)
    cuda_add_library(imScalingCUDA ../imageTransformation/imScaling.cu)
    # Color transformation
    cuda_compile(hueShift ../colorTransformation/hueShift.cu)
    cuda_add_library(hueShiftCUDA ../colorTransformation/hueShift.cu)
    # Image arithmetic
    cuda_compile(imArithmetic ../imageArithmetic/imArithmetic.cu)
    cuda_add_library(imArithmeticCUDA ../imageArithmetic/imArithmetic.cu)
    # Separable Gaussian filter
    cuda_compile(separableGauss ../separableGauss/separableGauss.cu)
    cuda_add_library(separableGaussCUDA ../separableGauss/separableGauss.cu)
    # Median filter
    cuda_compile(medianFilter ../medianFilter/medianFilter.cu)
    cuda_add_library(medianFilterCUDA ../medianFilter/medianFilter.cu)
    add_definitions(-DGPU_OPENCV_ENABLE)
endif()
  
# Link opencv libs to all executables
link_libraries(${OpenCV_LIBS})

# Create executables
# Gaussian filter
add_executable(gaussFilter ../convolutionGauss/gaussFilter.cpp)
target_link_libraries(gaussFilter gaussCUDA)
# Laplacian filter
add_executable(laplaceFilter ../convolutionLaplace/laplaceFilter.cpp)
target_link_libraries(laplaceFilter laplaceCUDA)
# Image scaling
add_executable(imScaling ../imageTransformation/imScaling.cpp)
target_link_libraries(imScaling imScalingCUDA)
# Color transformation
add_executable(hueShift ../colorTransformation/hueShift.cpp)
target_link_libraries(hueShift hueShiftCUDA)
# Image arithmetic
add_executable(imArithmetic ../imageArithmetic/imArithmetic.cpp)
target_link_libraries(imArithmetic imArithmeticCUDA)
# Separable Gaussian filter
add_executable(separableGauss ../separableGauss/separableGauss.cpp)
target_link_libraries(separableGauss separableGaussCUDA)
# Median filter
add_executable(medianFilter ../medianFilter/medianFilter.cpp)
target_link_libraries(medianFilter medianFilterCUDA)
