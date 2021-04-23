#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

__device__ float3 uchar3_to_float3(uchar3 elem) {
    float3 result;
    result.x = (float)elem.x;
    result.y = (float)elem.y;
    result.z = (float)elem.z;

    return result;
}

/**
 * Returns the linear interpolation of a and b by a factor of t.
 * Performs the following operation:
 * result = a + t*(b - a)
 *
 * @param a: first value to interpolate.
 * @param b: second value to interpolate.
 * @param t: interpolation factor.
 * @return result of the interpolation.
 */
__device__ float lerp(float a, float b, float t) { return a + t * (b - a); }

/**
 * Returns the bilinear interpolation of four pixels.
 * Performs the following operation:
 * inpH1 = lerp(pix1, pix2, intpX)
 * inpH2 = lerp(pix3, pix4, intpX)
 * inpV = lerp(inpH1, inpH2, intpY)
 *
 * @param pix1: first pixel to interpolate.
 * @param pix2: second pixel to interpolate.
 * @param pix3: third pixel to interpolate.
 * @param pix4: fourth pixel to interpolate.
 * @param intpX: horizontal interpolation factor.
 * @param intpY: vertical interpolation factor.
 * @return result of the bilinear interpolation.
 */
__device__ float3 interpolate_pix(float3 pix1, float3 pix2, float3 pix3,
				  float3 pix4, float intpX, float intpY) {
    // First horizontal interpolation
    float B_h1 = lerp(pix1.x, pix2.x, intpX);
    float G_h1 = lerp(pix1.y, pix2.y, intpX);
    float R_h1 = lerp(pix1.z, pix2.z, intpX);
    // Second horizontal interpolation
    float B_h2 = lerp(pix3.x, pix4.x, intpX);
    float G_h2 = lerp(pix3.y, pix4.y, intpX);
    float R_h2 = lerp(pix3.z, pix4.z, intpX);
    // Vertical interpolation
    float B = lerp(B_h1, B_h2, intpY);
    float G = lerp(G_h1, G_h2, intpY);
    float R = lerp(R_h1, R_h2, intpY);

    return make_float3(B, G, R);
}

__global__ void process(const cv::cuda::PtrStep<uchar3> src,
			cv::cuda::PtrStep<uchar3> dst, int rows, int cols,
			float scaleX, float scaleY) {

    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (dst_x < cols && dst_y < rows) {
	float x_pos, y_pos, intpX, intpY;
	intpX = modf(dst_x / scaleX, &x_pos);
	intpY = modf(dst_y / scaleY, &y_pos);
	float3 pix1 = uchar3_to_float3(src(y_pos, x_pos));
	float3 pix2 = uchar3_to_float3(src(y_pos, x_pos + 1));
	float3 pix3 = uchar3_to_float3(src(y_pos + 1, x_pos));
	float3 pix4 = uchar3_to_float3(src(y_pos + 1, x_pos + 1));

	float3 val = interpolate_pix(pix1, pix2, pix3, pix4, intpX, intpY);

	dst(dst_y, dst_x).x = val.x;
	dst(dst_y, dst_x).y = val.y;
	dst(dst_y, dst_x).z = val.z;
    }
}

int divUp(int a, int b) {
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void startCUDA (cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, float scaleX,
		float scaleY) {
    const dim3 block(32, 8);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    process<<<grid, block>>>(src, dst, dst.rows, dst.cols, scaleX, scaleY);
}

