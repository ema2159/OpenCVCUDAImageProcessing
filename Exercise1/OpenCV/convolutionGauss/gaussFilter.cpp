#include <chrono> // for high_resolution_clock
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <opencv2/opencv.hpp>

using namespace std;

/**
 * Returns the value of a 2-D Gaussian function with an standard
 * deviation of sigma at position (x, y)
 *
 * @param x: x value of the Gaussian function.
 * @param y: y value of the Gaussian function.
 * @param sigma: standard deviation of Gaussian function.
 * @return result of the 2-D Gaussian function.
 */
float gauss_2D(int x, int y, float sigma) {
  return exp(-((pow(x, 2.0) + pow(y, 2.0)) / (2.0 * pow(sigma, 2.0))));
}

/**
 * Returns a Gaussian kernel of size kernel_size x kernel_size and standard
 * deviation of sigma.
 *
 * @param kernel_size: size of the Gaussian kernel.
 * @param sigma: standard deviation of Gaussian function.
 * @return pointer to dynamic 2D array containing the Gaussian kernel.
 */
float **gauss_mat(int kernel_size, float sigma) {
  // Allocate space for the kernel
  float **kernel = new float *[kernel_size];
  for (int i = 0; i < kernel_size; i++) {
    kernel[i] = new float[kernel_size];
  }
  const int kernel_div_2 = kernel_size / 2;
  // Generate gaussian kernel of size kernel_size
  for (int m = -kernel_div_2; m <= kernel_div_2; m++) {
    for (int n = -kernel_div_2; n <= kernel_div_2; n++) {
      kernel[m + kernel_div_2][n + kernel_div_2] = gauss_2D(m, n, sigma);
    }
  }

  return kernel;
}

int main(int argc, char **argv) {

  cv::Mat source = cv::imread(argv[1], cv::IMREAD_COLOR);
  cv::Mat destination(source.rows, source.cols, CV_8UC3, cv::Scalar(0, 255, 0));

  cv::imshow("Source Image", source);

  auto begin = chrono::high_resolution_clock::now();

  const int KERNEL_SIZE = atoi(argv[2]);
  const int KERNEL_DIV_2 = KERNEL_SIZE / 2;
  const float SIGMA = 5;

  // Create input image which corresponds to the source image with an added
  // replication padding.
  cv::Mat input;
  cv::copyMakeBorder(source, input, KERNEL_DIV_2, KERNEL_DIV_2, KERNEL_DIV_2,
                     KERNEL_DIV_2, cv::BORDER_REPLICATE);

  // Create a gaussian kernel instead of calculating it each Time
  float **gauss_kernel = gauss_mat(KERNEL_SIZE, SIGMA);
  // Calculate sum of terms in the kernel for normalization
  float gauss_sum = 0;
  for (int i = 0; i < KERNEL_SIZE; i++) {
    for (int j = 0; j < KERNEL_SIZE; j++) {
      gauss_sum += gauss_kernel[i][j];
    }
  }

#pragma omp parallel for
  // The KERNEL_DIV_2 start is to account for the added padding to the input
  // image.
  for (int i = KERNEL_DIV_2; i < source.rows + KERNEL_DIV_2; i++) {
    // #pragma omp parallel for
    for (int j = KERNEL_DIV_2; j < source.cols + KERNEL_DIV_2; j++) {
      cv::Vec3f av = cv::Vec3f(0, 0, 0);
      for (int m = i - KERNEL_DIV_2; m <= i + KERNEL_DIV_2; m++) {
        for (int n = j - KERNEL_DIV_2; n <= j + KERNEL_DIV_2; n++) {
          av += input.at<cv::Vec3b>(m, n) *
                gauss_kernel[m - i + KERNEL_DIV_2][n - j + KERNEL_DIV_2];
        }
      }
      av /= gauss_sum;
      destination.at<cv::Vec3b>(i - KERNEL_DIV_2, j - KERNEL_DIV_2) = av;
    }
  }

  // Delete allocated memory
  for (int i = 0; i < KERNEL_SIZE; i++) {
    delete[] gauss_kernel[i];
  }
  delete[] gauss_kernel;

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - begin;

  cv::imshow("Processed Image", destination);

  cout << "Processing time: " << diff.count() << " s" << endl;

  cv::waitKey();
  return 0;
}
