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
 * Returns the value of a 2-D Gaussian function with an standard
 * deviation of sigma at position (x, y)
 *
 * @param x: x value of the Gaussian function.
 * @param y: y value of the Gaussian function.
 * @param sigma: standard deviation of Gaussian function.
 * @return result of the 2-D Gaussian function.
 */
__device__ float gauss_2D(int x, int y, float sigma) {
  return exp(-((pow(x, 2.0) + pow(y, 2.0)) / (2.0 * pow(sigma, 2.0))));
}

__global__ void process(const cv::cuda::PtrStep<uchar3> src,
			cv::cuda::PtrStep<uchar3> dst, int rows, int cols,
			int kernel_size, int sigma) {

    const int dst_x = (blockDim.x-kernel_size) * blockIdx.x + threadIdx.x;
    const int dst_y = (blockDim.y-kernel_size) * blockIdx.y + threadIdx.y;

    // Filter radius
    const int kernel_div2 = kernel_size / 2;

    // Create shared memory using externally passed size
    extern __shared__ uchar3 tile[];

    int px = dst_x;
    int py = dst_y;

    // Cache pixels in shared memory
    tile[threadIdx.y*(TILE_SIZE+kernel_size)+threadIdx.x] = src(py, cols+px);

    // Wait until all thread cache their pixes values
    __syncthreads();  

    bool is_inside_tile =
	kernel_div2 <= threadIdx.x && threadIdx.x < TILE_SIZE + kernel_div2 &&
	kernel_div2 <= threadIdx.y && threadIdx.y < TILE_SIZE + kernel_div2;
    if (dst_x < cols-kernel_div2 && dst_y < rows-kernel_div2 && is_inside_tile) {
	float3 val = make_float3(0, 0, 0);
	float gauss_sum = 0;
	for (int m = -kernel_div2; m <= kernel_div2; m++) {
	    for (int n = -kernel_div2; n <= kernel_div2; n++) {
		float gauss_val = gauss_2D(m, n, sigma);
		gauss_sum += gauss_val;

		int ty = threadIdx.y+n;
		int tx = threadIdx.x+m;
		uchar3 pix = tile[ty*(TILE_SIZE+kernel_size)+tx];
		val.x += (float)pix.z*gauss_val;
		val.y += (float)pix.x*gauss_val;
		val.z += (float)pix.y*gauss_val;
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
		float SIGMA ) {
  const dim3 block(TILE_SIZE+KERNEL_SIZE, TILE_SIZE+KERNEL_SIZE);
  const dim3 grid(divUp(src.cols, block.x), divUp(src.rows, block.y));

  
  // Create a tile to process pixels within a block's shared memory
  int shmem_size = sizeof(uchar3)*(TILE_SIZE+KERNEL_SIZE)*(TILE_SIZE+KERNEL_SIZE);
  // printf("AAAAAAAAAAAAAAA %i\n", TILE_SIZE);
  // printf("AAAAAAAAAAAAAAA %i\n", KERNEL_SIZE);
  // printf("AAAAAAAAAAAAAAA %i\n", shmem_size);
  
  process<<<grid, block, shmem_size>>>(src, dst, src.rows, src.cols, KERNEL_SIZE, SIGMA);

}

