#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

__device__ float lerp(float a, float b, float t) { return a + t * (b - a); }

float3 interpolate_pix(float3 pix1, float3 pix2, float3 pix3,
                          float3 pix4, float intpX, float intpY) {
  // First horizontal interpolation
  float B_h1 = lerp(pix1[0], pix2[0], intpX);
  float G_h1 = lerp(pix1[1], pix2[1], intpX);
  float R_h1 = lerp(pix1[2], pix2[2], intpX);
  // Second horizontal interpolation
  float B_h2 = lerp(pix3[0], pix4[0], intpX);
  float G_h2 = lerp(pix3[1], pix4[1], intpX);
  float R_h2 = lerp(pix3[2], pix4[2], intpX);
  // Vertical interpolation
  float B = lerp(B_h1, B_h2, intpY);
  float G = lerp(G_h1, G_h2, intpY);
  float R = lerp(R_h1, R_h2, intpY);

  return make_float3(B, G, R);
}

__global__ void process(const cv::cuda::PtrStep<uchar3> src,
			cv::cuda::PtrStep<uchar3> dst, int rows, int cols) {

    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (dst_x < cols && dst_y < rows) {
	const int kernel_div2 = 1;
	// Generate gaussian kernel of size kernel_size
	float3 val = make_float3(0, 255, 0);

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

