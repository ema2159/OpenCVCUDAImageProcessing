#include <chrono> // for high_resolution_clock
#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>

using namespace std;

/**
 * Returns the value of a 2-D Gaussian function with an standard
 * deviation of sigma at position (x, y)
 *
 * @param x: x position of pixel in the kernel.
 * @param y: y position of pixel in the kernel.
 * @param sigma: standard deviation of gaussian kernel.
 * @return result of the 2-D gaussian function.
 */
float get_gauss_pix(int x, int y, float sigma) {
  return exp(-((pow(x, 2.0) + pow(y, 2.0)) / (2.0 * pow(sigma, 2.0))));
}

int main(int argc, char **argv) {

  cv::Mat source = cv::imread(argv[1], cv::IMREAD_COLOR);
  cv::Mat destination(source.rows, source.cols, CV_8UC3, cv::Scalar(0, 255, 0));

  cv::imshow("Source Image", source);

  cout << "Threads: " << omp_get_num_threads() << endl;

  auto begin = chrono::high_resolution_clock::now();
  const int iter = 1;

  const unsigned int KERNEL_SIZE = 19;
  const unsigned int KERNEL_DIV_2 = KERNEL_SIZE / 2;
  const float SIGMA = 5;

  // Create input image which corresponds to the source image with an added replication padding.
  cv::Mat input;
  cv::copyMakeBorder(source, input, KERNEL_DIV_2, KERNEL_DIV_2, KERNEL_DIV_2,
                     KERNEL_DIV_2, cv::BORDER_REPLICATE);

  for (int it = 0; it < iter; it++) {
#pragma omp parallel for
    // The KERNEL_DIV_2 start is to accout for the added padding to the input image.
    for (int i = KERNEL_DIV_2; i < source.rows+KERNEL_DIV_2; i++) {
      // #pragma omp parallel for
      for (int j = KERNEL_DIV_2; j < source.cols+KERNEL_DIV_2; j++) {
        cv::Vec3f av = cv::Vec3f(0, 0, 0);
        float kernel_sum = 0;
        float curr_gauss;
        for (int m = i - (KERNEL_SIZE / 2); m <= i + (KERNEL_SIZE / 2); m++) {
          for (int n = j - (KERNEL_SIZE / 2); n <= j + (KERNEL_SIZE / 2); n++) {
            curr_gauss = get_gauss_pix(m - i, n - j, SIGMA);
            kernel_sum += curr_gauss;
            av += input.at<cv::Vec3b>(m, n) * curr_gauss;
          }
        }
        av /= kernel_sum;
        destination.at<cv::Vec3b>(i-KERNEL_DIV_2, j-KERNEL_DIV_2) = av;
      }
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - begin;

  cv::imshow("Processed Image", destination);

  cout << "Total time: " << diff.count() << " s" << endl;
  cout << "Time for 1 iteration: " << diff.count() / iter << " s" << endl;
  cout << "IPS: " << iter / diff.count() << endl;

  cv::waitKey();
  return 0;
}
