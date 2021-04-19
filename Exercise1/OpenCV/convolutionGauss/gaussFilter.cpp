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
  cv::Mat destination (source.rows, source.cols, CV_8UC3, cv::Scalar(0, 255, 0));
  
  cv::imshow("Source Image", source);

  cv::imshow("Processed Image", destination);

  cv::waitKey();
  return 0;
}
