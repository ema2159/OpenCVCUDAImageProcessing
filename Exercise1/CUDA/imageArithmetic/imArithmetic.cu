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
 * Clamps a vector's values between low and high.
 *
 * @param elem: element to clamp.
 * @param low: lower bound for the clamp.
 * @param high: upper bound for the clamp.
 * @return a vector with the original vector values clamped in the specified interval.
 */
__device__ float3 clamp(float3 elem, float low, float high) {
    float3 result;
    result.x = max(low, min(high, elem.x));
    result.y = max(low, min(high, elem.y));
    result.z = max(low, min(high, elem.z));

    return result;
}

/**
 * Do an arithmetic operation between two pixels.
 * I does the following opertion:
 *
 * (pix1 (operation) pix2) * scale_factor + offset
 *
 * @param pix1: corresponds to the first pixel argument.
 * @param pix2: corresponds to the second pixel argument.
 * @param operation: the operator to utilize. Choose between +, -, * and /
 * @param scale_factor: scale factor for the pixel operation.
 * @param offest: offset for the pixel operation.
 * @return pixel that results from the operation.
 */
__device__ float3 pix_arithm(float3 pix1, float3 pix2, char operation,
			     float scale_factor, float offset) {
    float3 result;
    float3 vec_offset = make_float3(offset, offset, offset);
    switch (operation) {
    case '+': {
	result.x = pix1.x + pix2.x;
	result.y = pix1.y + pix2.y;
	result.z = pix1.z + pix2.z;
	break;
    }
    case '-': {
	result.x = pix1.x - pix2.x;
	result.y = pix1.y - pix2.y;
	result.z = pix1.z - pix2.z;
	break;
    }
    case 'x': {
	result.x = pix1.x * pix2.x;
	result.y = pix1.y * pix2.y;
	result.z = pix1.z * pix2.z;
	break;
    }
    case '/': {
	result.x = pix1.x / pix2.x;
	result.y = pix1.y / pix2.y;
	result.z = pix1.z / pix2.z;
	break;
    }
    default:
	result = make_float3(0, 0, 0);
	break;
    }
    result.x = result.x * scale_factor;
    result.y = result.y * scale_factor;
    result.z = result.z * scale_factor;    

    result.x = result.x + vec_offset.x;
    result.y = result.y + vec_offset.y;
    result.z = result.z + vec_offset.z;
    return result;
}

__global__ void process(const cv::cuda::PtrStep<uchar3> src1,
			const cv::cuda::PtrStep<uchar3> src2,
			cv::cuda::PtrStep<uchar3> dst, int rows, int cols,
			char operation, float scale_factor, float offset) {

    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (dst_x < cols && dst_y < rows) {
	float3 val = pix_arithm(uchar3_to_float3(src1(dst_y, dst_x)),
				uchar3_to_float3(src2(dst_y, dst_x)),
				operation, scale_factor, offset);

	val = clamp(val, 0, 255);

	dst(dst_y, dst_x).x = val.x;
	dst(dst_y, dst_x).y = val.y;
	dst(dst_y, dst_x).z = val.z;
    }
}

int divUp(int a, int b) {
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void startCUDA (cv::cuda::GpuMat& src1, cv::cuda::GpuMat& src2,
		cv::cuda::GpuMat& dst, char operation, float scale_factor,
		float offset) {
    const dim3 block(32, 8);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    process<<<grid, block>>>(src1, src2, dst, dst.rows, dst.cols, operation,
			     scale_factor, offset);
}
