#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>
#include <thrust/device_vector.h>

#define MAX_ARRAY_SIZE 400
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
 * Swaps two elements of an array.
 * @param a: Memory location of first element.
 * @param b: Memory location of second element.
 */
template< typename T > __device__ void swap(T* a, T* b) {
    T temp = *a;
    *a = *b;
    *b = temp;
    return;
}

/**
 * Returns the length of a vector.
 *
 * @param elem: the vector to obtain the length from.
 * @return the length of the given vector.
 */
template< typename T > __device__ float length(T elem) {
    return sqrt(pow((float)elem.x, 2) + pow((float)elem.y, 2)
		+ pow((float)elem.z, 2));
}

/**
 * Implementation of a standard Lomuto partiton algorithm.
 *
 * @param arr: the array to partition.
 * @param low: starting point for the partition in the array.
 * @param high: ending point for the partition in the array.
 * @return the position of the pivot.
 */
template< typename T > __device__ int partition(T arr[], int low, int high) {
    T pivot = arr[high];
    int i = (low - 1);
    for (int j = low; j <= high - 1; j++) {
        if (length<T>(arr[j]) <= length<T>(pivot)) {
            i++;
            swap<T>(&arr[i], &arr[j]);
        }
    }
    swap<T>(&arr[i + 1], &arr[high]);
    return (i + 1);
}
  
/**
 * Returns the kth smallest element from a given array.
 *
 * @param arr: the array to find the smallest element from.
 * @param left: starting point for the search in the array.
 * @param right: ending point for the search in the array.
 * @param k: value corresponding to the position of the element to find in the 
 * original array when sorted.
 * @return the kth smallest element in the array.
 */
template< typename T > __device__ T kth_smallest(T a[], int left, int right,
						 int k) {
  
    while (left <= right) {
  
        // Partition a[left..right] around a pivot
        // and find the position of the pivot
        int pivotIndex = partition<T>(a, left, right);
  
        // If pivot itself is the k-th smallest element
        if (pivotIndex == k - 1)
            return a[pivotIndex];
  
        // If there are more than k-1 elements on
        // left of pivot, then k-th smallest must be
        // on left side.
        else if (pivotIndex > k - 1)
            right = pivotIndex - 1;
  
        // Else k-th smallest is on right side.
        else
            left = pivotIndex + 1;
    }
    return a[0];
}

__global__ void process(const cv::cuda::PtrStep<uchar3> src,
                        cv::cuda::PtrStep<uchar3> dst, int rows, int cols,
                        int kernel_size) {

    const int dst_x = TILE_SIZE * blockIdx.x + threadIdx.x-kernel_size;
    const int dst_y = TILE_SIZE * blockIdx.y + threadIdx.y-kernel_size;

    // Filter radius
    const int kernel_div2 = kernel_size / 2;

    // Create shared memory using externally passed size
    extern __shared__ uchar3 tile[];

    int px = clamp<float>(dst_x, 0, cols-1);
    int py = clamp<float>(dst_y, 0, rows-1);

    // Cache pixels in shared memory
    tile[threadIdx.y*(TILE_SIZE+kernel_size)+threadIdx.x] = src(py, px);

    // Wait until all thread cache their pixes values
    __syncthreads();  

    bool is_inside_tile =
        kernel_div2 <= threadIdx.x && threadIdx.x < TILE_SIZE + kernel_div2 &&
        kernel_div2 <= threadIdx.y && threadIdx.y < TILE_SIZE + kernel_div2;
    if (dst_x < cols && dst_y < rows && is_inside_tile) {
	uchar3 vals[MAX_ARRAY_SIZE];
	int count = 0;
	for (int m = -kernel_div2; m <= kernel_div2; m++) {
	    for (int n = -kernel_div2; n <= kernel_div2; n++) {
                int ty = threadIdx.y+n;
                int tx = threadIdx.x+m;
		vals[count] = tile[ty*(TILE_SIZE+kernel_size)+tx];
		count++;
	    }
	}

	int arr_size = (int)pow(kernel_size, 2);
	uchar3 median = kth_smallest<uchar3>(vals, 0, arr_size, arr_size/2);

	dst(dst_y, dst_x).x = median.x;
	dst(dst_y, dst_x).y = median.y;
	dst(dst_y, dst_x).z = median.z;
    }
}

int divUp(int a, int b) {
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void startCUDA (cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int KERNEL_SIZE) {
  const dim3 block(TILE_SIZE+KERNEL_SIZE, TILE_SIZE+KERNEL_SIZE);
  const dim3 grid(divUp(dst.cols, TILE_SIZE)+1, divUp(dst.rows, TILE_SIZE)+1);

  
  // Create a tile to process pixels within a block's shared memory
  int shmem_size = sizeof(uchar3)*(TILE_SIZE+KERNEL_SIZE)*(TILE_SIZE+KERNEL_SIZE);
  
  process<<<grid, block, shmem_size>>>(src, dst, dst.rows, dst.cols, KERNEL_SIZE);

}