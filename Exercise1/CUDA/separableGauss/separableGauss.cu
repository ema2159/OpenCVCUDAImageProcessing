#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

/**
 * Returns the value of a Gaussian function with an standard
 * deviation of sigma and an input value of x.
 *
 * @param x: input value of the Gaussian function.
 * @param sigma: standard deviation of Gaussian function.
 * @return result of the Gaussian function.
 */
__device__ float gauss_func(int x, float sigma) {
  return exp(-(pow(x, 2.0) / (2.0 * pow(sigma, 2.0))));
}

__global__ void process(const cv::cuda::PtrStep<uchar3> src,
			cv::cuda::PtrStep<uchar3> dst, int rows, int cols,
			int kernel_size, int sigma, bool first_pass) {

    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (dst_x < cols && dst_y < rows) {
	const int kernel_div2 = kernel_size / 2;
	// Generate gaussian kernel of size kernel_size
	float3 val = make_float3(0, 0, 0);
	float gauss_sum = 0;
	if (first_pass) {
	    for (int m = -kernel_div2; m <= kernel_div2; m++) {
		float gauss_val = gauss_func(m, sigma);
		gauss_sum += gauss_val;
		val.x += (float)src(dst_y+kernel_div2, dst_x+kernel_div2+m).x*gauss_val;
		val.y += (float)src(dst_y+kernel_div2, dst_x+kernel_div2+m).y*gauss_val;
		val.z += (float)src(dst_y+kernel_div2, dst_x+kernel_div2+m).z*gauss_val;
	    }
	}
	else {
	    for (int n = -kernel_div2; n <= kernel_div2; n++) {
		float gauss_val = gauss_func(n, sigma);
		gauss_sum += gauss_val;
		val.x += (float)src(dst_y+kernel_div2+n, dst_x+kernel_div2).x*gauss_val;
		val.y += (float)src(dst_y+kernel_div2+n, dst_x+kernel_div2).y*gauss_val;
		val.z += (float)src(dst_y+kernel_div2+n, dst_x+kernel_div2).z*gauss_val;
	    }
	}
	val.x = val.x/gauss_sum;
	val.y = val.y/gauss_sum;
	val.z = val.z/gauss_sum;

	dst(dst_y, dst_x).x = val.x;
	dst(dst_y, dst_x).y = val.y;
	dst(dst_y, dst_x).z = val.z;
    }
}

int divUp(int a, int b) {
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void startCUDA (cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int KERNEL_SIZE,
		 float SIGMA, bool first_pass) {
  const dim3 block(32, 8);
  const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

  process<<<grid, block>>>(src, dst, dst.rows, dst.cols, KERNEL_SIZE, SIGMA,
			   first_pass);
}

