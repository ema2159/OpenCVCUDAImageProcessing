#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

#define TILE_SIZE 15

/**
 * Clamps the value val in the interval [lo, high].
 * Equivalent to max(lo, min(val, high)).
 *
 * @param val: value to clamp.
 * @param lo: lower bound for the clamping.
 * @param high: higher bound for the clamping.
 * @return val clamped between lo and high.
 */
template< typename T > __device__ T clamp(T val, T lo, T high) {
  return max(lo, min(val, high));
}

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

    const int dst_x = TILE_SIZE * blockIdx.x + threadIdx.x-kernel_size;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    // Filter radius
    const int kernel_div2 = kernel_size / 2;

    // Create shared memory using externally passed size
    extern __shared__ uchar3 tile[];

    int px = clamp<float>(dst_x, 0, cols-1);
    int py = clamp<float>(dst_y, 0, rows-1);

    // Cache pixels in shared memory
    tile[threadIdx.x] = src(py, px);

    // Wait until all thread cache their pixes values
    __syncthreads();  

    bool is_inside_tile =
        kernel_div2 <= threadIdx.x && threadIdx.x < TILE_SIZE + kernel_div2;
    if (dst_x < cols && dst_y < rows && is_inside_tile) {
	float3 val = make_float3(0, 0, 0);
	float gauss_sum = 0;
	for (int m = -kernel_div2; m <= kernel_div2; m++) {
	    float gauss_val = gauss_func(m, sigma);
	    gauss_sum += gauss_val;

	    int tx = threadIdx.x+m;
	    uchar3 pix = tile[tx];
	    val.x += (float)pix.x*gauss_val;
	    val.y += (float)pix.y*gauss_val;
	    val.z += (float)pix.z*gauss_val;
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
  const dim3 block(TILE_SIZE+KERNEL_SIZE);
  const dim3 grid(divUp(dst.cols, TILE_SIZE)+1, divUp(dst.rows, block.y));

  
  // Create a tile to process pixels within a block's shared memory
  int shmem_size = sizeof(uchar3)*(TILE_SIZE+KERNEL_SIZE);
  
  process<<<grid, block, shmem_size>>>(src, dst, dst.rows, dst.cols,
				       KERNEL_SIZE, SIGMA, first_pass);

}

