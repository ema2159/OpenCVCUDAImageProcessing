#include <chrono> // for high_resolution_clock
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

int clamp(int val, int lo, int high) {
  return std::max(lo, std::min(val, high));
}

int main(int argc, char **argv) {

  cv::Mat source = cv::imread(argv[1], cv::IMREAD_COLOR);
  cv::Mat destination (source.rows, source.cols, CV_8UC3, cv::Scalar(0, 255, 0));
  
  cv::imshow("Source Image", source);

  cv::imshow("Processed Image", destination);

  cv::waitKey();
  return 0;
}
