cmake_minimum_required(VERSION 2.8)
project(CUDAOpenCVImageProcessing)

# Include OpenCV
find_package(OpenCV REQUIRED PATHS "/home/ema2159/.local/opencv/opencv-4.2.0/build")
include_directories(${OpenCV_INCLUDE_DIRS})

# Include CUDA
find_package(CUDA REQUIRED)
# find_package(CUDA REQUIRED)
# if(CUDA_FOUND)
#     SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode arch=compute_20,code=sm_20)
#     cuda_compile(CudaKernel CudaKernels.cu)
#     cuda_add_library(CudaKernels CudaKernels.cu)
#     add_definitions(-DGPU_OPENCV_ENABLE)
# endif()
  
# Include OpenMP
find_package(OpenMP)

# Link libraries
link_libraries(${OpenCV_LIBS})
if(OpenMP_CXX_FOUND)
    link_libraries(OpenMP::OpenMP_CXX)
endif()

# Crete executables
add_executable( gaussFilter ../convolutionGauss/gaussFilter.cpp )
add_executable( laplacianFilter ../convolutionLaplace/laplacianFilter.cpp )
add_executable( imArithmetic ../imageArithmetic/imArithmetic.cpp )
add_executable( sepGaussFilter ../separableGauss/sepGaussFilter.cpp )
add_executable( medianFilter ../medianFilter/medianFilter.cpp )
