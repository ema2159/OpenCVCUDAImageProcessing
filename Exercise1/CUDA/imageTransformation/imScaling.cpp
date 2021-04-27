#include <chrono> // for high_resolution_clock
#include <iostream>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>

using namespace std;

void startCUDA(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst, float scaleX, float scaleY);

int main(int argc, char **argv) {
  cv::namedWindow("Original Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Processed Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);

  cv::Mat h_img = cv::imread(argv[1]);
  float scaleX = atof(argv[2]);
  float scaleY = atof(argv[3]);
  cv::Mat h_result(h_img.rows*scaleY, h_img.cols*scaleX, CV_8UC3, cv::Scalar(0, 255, 0));
  cv::cuda::GpuMat d_img, d_result;

  cv::imshow("Original Image", h_img);

  auto begin = chrono::high_resolution_clock::now();
  const int iter = 1;

  d_img.upload(h_img);
  d_result.upload(h_result);

  for (int i = 0; i < iter; i++) {
    startCUDA(d_img, d_result, scaleX, scaleY);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - begin;

  cv::imshow("Processed Image", d_result);

  cout << diff.count() << endl;
  cout << diff.count() / iter << endl;
  cout << iter / diff.count() << endl;

  cv::waitKey();
  return 0;
}
