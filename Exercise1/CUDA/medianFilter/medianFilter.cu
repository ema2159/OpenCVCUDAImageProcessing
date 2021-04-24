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

template< typename T > __device__ void swap(T* a, T* b) {
    T temp = *a;
    *a = *b;
    *b = temp;
    return;
}

template< typename T > __device__ float length(T elem) {
    return sqrt(pow((float)elem.x, 2) + pow((float)elem.y, 2)
		+ pow((float)elem.z, 2));
}

// Standard Lomuto partition function
__device__ int partition(uchar3 arr[], int low, int high) {
    uchar3 pivot = arr[high];
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
  
// Implementation of QuickSelect
__device__ uchar3 kth_smallest(uchar3 a[], int left, int right, int k) {
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
			const int kernel_size) {

    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (dst_x < cols && dst_y < rows) {
	const int kernel_div2 = kernel_size / 2;
	uchar3 vals[MAX_ARRAY_SIZE];
	int count = 0;
	for (int m = -kernel_div2; m <= kernel_div2; m++) {
	    for (int n = -kernel_div2; n <= kernel_div2; n++) {
		vals[count] = src(dst_y+n, dst_x+m);
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
  const dim3 block(32, 8);
  const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

  process<<<grid, block>>>(src, dst, dst.rows, dst.cols, KERNEL_SIZE);

}

