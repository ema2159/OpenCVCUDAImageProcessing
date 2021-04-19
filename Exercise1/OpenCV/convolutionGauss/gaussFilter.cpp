#include <chrono> // for high_resolution_clock
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;


/**
 * Clamps the value val in the interval [lo, high].
 * Equivalent to max(lo, min(val, high)).
 *
 * @param val: value to clamp.
 * @param lo: lower bound for the clamping.
 * @param high: higher bound for the clamping.
 * @return val clamped between lo and high.
 */
int clamp(int val, int lo, int high) {
  return max(lo, min(val, high));
}

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

  auto begin = chrono::high_resolution_clock::now();
  const int iter = 1;

  const unsigned int KERNEL_SIZE = 19;
  const float SIGMA = 3;

  for (int it = 0; it < iter; it++) {
#pragma omp parallel for
    for (int i = 0; i < source.rows; i++) {
      // #pragma omp parallel for
      for (int j = 0; j < source.cols; j++) {
        cv::Vec3f av = cv::Vec3f(0, 0, 0);
	float kernel_sum = 0;
	float curr_gauss;
        for (int m = i-(KERNEL_SIZE / 2); m <= i+(KERNEL_SIZE / 2); m++) {
          for (int n = j-(KERNEL_SIZE / 2); n <= j+(KERNEL_SIZE / 2); n++) {
	    curr_gauss = get_gauss_pix(m-i, n-j, SIGMA);
	    kernel_sum += curr_gauss;
            av += source.at<cv::Vec3b>(clamp(m, 0, source.rows), clamp(n, 0, source.cols))*curr_gauss;
          }
        }
        av /= kernel_sum;
        destination.at<cv::Vec3b>(i, j) = av;
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
