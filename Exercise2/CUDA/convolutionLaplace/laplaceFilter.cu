#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

#define TILE_SIZE 28
#define KERNEL_SIZE 3
#define KERNEL_DIV2 1

__device__ const float LAPLACE_KERNEL[3][3] = {{-1,-1,-1},
					       {-1, 8,-1},
					       {-1,-1,-1}};

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

__global__ void process(const cv::cuda::PtrStep<uchar3> src,
			cv::cuda::PtrStep<uchar3> dst, int rows, int cols) {

    const int dst_x = TILE_SIZE * blockIdx.x + threadIdx.x-KERNEL_SIZE;
    const int dst_y = TILE_SIZE * blockIdx.y + threadIdx.y-KERNEL_SIZE;

    // Create shared memory using externally passed size
    extern __shared__ uchar3 tile[];

    int px = clamp<int>(dst_x, 0, cols-1);
    int py = clamp<int>(dst_y, 0, rows-1);

    // Cache pixels in shared memory
    tile[threadIdx.y*(TILE_SIZE+KERNEL_SIZE)+threadIdx.x] = src(py, px);

    // Wait until all thread cache their pixes values
    __syncthreads();  

    bool is_inside_tile =
        KERNEL_DIV2 <= threadIdx.x && threadIdx.x < TILE_SIZE + KERNEL_DIV2 &&
        KERNEL_DIV2 <= threadIdx.y && threadIdx.y < TILE_SIZE + KERNEL_DIV2;
    if (dst_x < cols && dst_y < rows && is_inside_tile) {
	float3 val = make_float3(0, 0, 0);
	float gauss_sum = 0;
        for (int m = -KERNEL_DIV2; m <= KERNEL_DIV2; m++) {
            for (int n = -KERNEL_DIV2; n <= KERNEL_DIV2; n++) {
                int ty = threadIdx.y+n;
                int tx = threadIdx.x+m;
                uchar3 pix = tile[ty*(TILE_SIZE+KERNEL_SIZE)+tx];
		val.x += (float)pix.x
		    *LAPLACE_KERNEL[m+KERNEL_DIV2][n+KERNEL_DIV2];
		val.y += (float)pix.y
		    *LAPLACE_KERNEL[m+KERNEL_DIV2][n+KERNEL_DIV2];
		val.z += (float)pix.z
		    *LAPLACE_KERNEL[m+KERNEL_DIV2][n+KERNEL_DIV2];
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
    const dim3 block(TILE_SIZE+KERNEL_SIZE, TILE_SIZE+KERNEL_SIZE);
    const dim3 grid(divUp(dst.cols, TILE_SIZE)+1, divUp(dst.rows, TILE_SIZE)+1);

  
    // Create a tile to process pixels within a block's shared memory
    int shmem_size = sizeof(uchar3)*(TILE_SIZE+KERNEL_SIZE)*(TILE_SIZE+KERNEL_SIZE);
  
    process<<<grid, block, shmem_size>>>(src, dst, dst.rows, dst.cols);
}