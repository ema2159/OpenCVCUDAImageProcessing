#include <chrono> // for high_resolution_clock
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <opencv2/opencv.hpp>

using namespace std;

/**
 * Returns the value of a Gaussian function with an standard
 * deviation of sigma with an input value x
 *
 * @param x: input value of the Gaussian function.
 * @param sigma: standard deviation of Gaussian function.
 * @return result of the Gaussian function.
 */
float gauss_func(int x, float sigma) {
  return exp(-(pow(x, 2.0) / (2.0 * pow(sigma, 2.0))));
}

/**
 * Returns a Gaussian 1-D kernel of size kernel_size and standard
 * deviation of sigma.
 *
 * @param kernel_size: size of the Gaussian kernel.
 * @param sigma: standard deviation of Gaussian function.
 * @return pointer to dynamic array containing the 1-D Gaussian kernel.
 */
float *gauss_mat(int kernel_size, float sigma) {
  // Allocate space for the kernel
  float *kernel = new float[kernel_size];

  const int kernel_div_2 = kernel_size / 2;
  // Generate gaussian kernel of size kernel_size
  for (int m = -kernel_div_2; m <= kernel_div_2; m++) {
    kernel[m + kernel_div_2] = gauss_func(m, sigma);
  }

  return kernel;
}

int main(int argc, char **argv) {

  cv::Mat source = cv::imread(argv[1], cv::IMREAD_COLOR);
  cv::Mat intermediate(source.rows, source.cols, CV_8UC3, cv::Scalar(0, 255, 0));
  cv::Mat destination(source.rows, source.cols, CV_8UC3, cv::Scalar(0, 255, 0));

  cv::imshow("Source Image", source);

  auto begin = chrono::high_resolution_clock::now();

  const unsigned int KERNEL_SIZE = 31;
  const unsigned int KERNEL_DIV_2 = KERNEL_SIZE / 2;
  const float SIGMA = 5;

  // Create input image which corresponds to the source image with an added
  // replication padding.
  cv::Mat input;
  cv::copyMakeBorder(source, input, KERNEL_DIV_2, KERNEL_DIV_2, KERNEL_DIV_2,
                     KERNEL_DIV_2, cv::BORDER_REPLICATE);

  // Create a gaussian kernel instead of calculating it each Time
  float *gauss_kernel = gauss_mat(KERNEL_SIZE, SIGMA);
  // Calculate sum of terms in the kernel for normalization
  float gauss_sum = 0;
  for (int i = 0; i < KERNEL_SIZE; i++) {
    gauss_sum += gauss_kernel[i];
  }

#pragma omp parallel for
  // The KERNEL_DIV_2 start is to account for the added padding to the input
  // image.
  for (int i = KERNEL_DIV_2; i < source.rows + KERNEL_DIV_2; i++) {
    // #pragma omp parallel for
    for (int j = KERNEL_DIV_2; j < source.cols + KERNEL_DIV_2; j++) {
      cv::Vec3f av = cv::Vec3f(0, 0, 0);
      for (int m = i - KERNEL_DIV_2; m <= i + KERNEL_DIV_2; m++) {
	av += input.at<cv::Vec3b>(m, j) *
	  gauss_kernel[m - i + KERNEL_DIV_2];
      }
      av /= gauss_sum;
      intermediate.at<cv::Vec3b>(i - KERNEL_DIV_2, j - KERNEL_DIV_2) = av;
    }
  }

  // Create a padded image for the intermediate image
  cv::copyMakeBorder(intermediate, input, KERNEL_DIV_2, KERNEL_DIV_2,
		     KERNEL_DIV_2, KERNEL_DIV_2, cv::BORDER_REPLICATE);
#pragma omp parallel for
  for (int i = KERNEL_DIV_2; i < source.rows + KERNEL_DIV_2; i++) {
    // #pragma omp parallel for
    for (int j = KERNEL_DIV_2; j < source.cols + KERNEL_DIV_2; j++) {
      cv::Vec3f av = cv::Vec3f(0, 0, 0);
      for (int n = j - KERNEL_DIV_2; n <= j + KERNEL_DIV_2; n++) {
	av += input.at<cv::Vec3b>(i, n) *
	      gauss_kernel[n - j + KERNEL_DIV_2];
      }
      av /= gauss_sum;
      destination.at<cv::Vec3b>(i - KERNEL_DIV_2, j - KERNEL_DIV_2) = av;
    }
  }

  // Delete allocated memory
  delete[] gauss_kernel;

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - begin;

  cv::imshow("Processed Image", destination);

  cout << "Processing time: " << diff.count() << " s" << endl;

  cv::waitKey();
  return 0;
}
