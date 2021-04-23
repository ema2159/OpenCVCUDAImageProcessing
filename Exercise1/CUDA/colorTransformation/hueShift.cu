#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

/**
 * Converts a uchar3 element to a float3.
 *
 * @param elem: element to convert.
 * @return input a float 3 with the input element fields casted to float.
 */
__device__ float3 uchar3_to_float3(uchar3 elem) {
    float3 result;
    result.x = (float)elem.x;
    result.y = (float)elem.y;
    result.z = (float)elem.z;

    return result;
}

/**
 * Shifts the hue of a pixel in the CIELAB colorspace.
 *
 * @param pix: pixel to transform, given in the CIELAB colorspace.
 * @param angle: angle to add to the pixel's hue angle.
 * @return hue shifter pixel.
 */
__device__ float3 hueShift(float3 pix, float angle) {
  float C_ab = sqrt(pow(pix.y, 2) + pow(pix.z, 2)); // sqrt((a*)^2+(b*)^2)
  float h = atan2(pix.z, pix.y);

  // Shift CIELAB color's a* and b* components
  h += (angle * M_PI) / 180.0;

  return make_float3(pix.x, cos(h) * C_ab, sin(h) * C_ab);
}

__global__ void process(const cv::cuda::PtrStep<uchar3> src,
			cv::cuda::PtrStep<uchar3> dst, int rows, int cols,
			float angle) {

    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (dst_x < cols && dst_y < rows) {

	float3 val = hueShift(uchar3_to_float3(src(dst_y, dst_x)), angle);

	dst(dst_y, dst_x).x = val.x;
	dst(dst_y, dst_x).y = val.y;
	dst(dst_y, dst_x).z = val.z;
    }
}

int divUp(int a, int b) {
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void startCUDA (cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, float angle) {
    const dim3 block(32, 8);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    process<<<grid, block>>>(src, dst, dst.rows, dst.cols, angle);
}

