cmake_minimum_required(VERSION 2.8)
project(CUDAOpenCVImageProcessing)

# Include OpenCV
find_package(OpenCV REQUIRED PATHS "/home/ema2159/.local/opencv/opencv-4.2.0/build")
include_directories(${OpenCV_INCLUDE_DIRS})
  
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
