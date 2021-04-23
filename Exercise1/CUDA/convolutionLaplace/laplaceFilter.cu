#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

__device__ const float LAPLACE_KERNEL[3][3] = {{-1,-1,-1},
					       {-1, 8,-1},
					       {-1,-1,-1}};

__global__ void process(const cv::cuda::PtrStep<uchar3> src,
			cv::cuda::PtrStep<uchar3> dst, int rows, int cols) {

    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (dst_x < cols && dst_y < rows) {
	const int kernel_div2 = 1;
	// Generate gaussian kernel of size kernel_size
	float3 val = make_float3(0, 0, 0);
	for (int m = -kernel_div2; m <= kernel_div2; m++) {
	    for (int n = -kernel_div2; n <= kernel_div2; n++) {
		val.x += (float)src(dst_y+kernel_div2+n, dst_x+kernel_div2+m).x
		    *LAPLACE_KERNEL[m+kernel_div2][n+kernel_div2];
		val.y += (float)src(dst_y+kernel_div2+n, dst_x+kernel_div2+m).y
		    *LAPLACE_KERNEL[m+kernel_div2][n+kernel_div2];
		val.z += (float)src(dst_y+kernel_div2+n, dst_x+kernel_div2+m).z
		    *LAPLACE_KERNEL[m+kernel_div2][n+kernel_div2];
	    }
	}

	dst(dst_y, dst_x).x = val.x;
	dst(dst_y, dst_x).y = val.y;
	dst(dst_y, dst_x).z = val.z;
    }
}

int divUp(int a, int b) {
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void startCUDA ( cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst) {
    const dim3 block(32, 8);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    process<<<grid, block>>>(src, dst, dst.rows, dst.cols);
}

