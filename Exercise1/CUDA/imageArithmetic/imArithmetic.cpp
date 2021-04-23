#include <chrono> // for high_resolution_clock
#include <iostream>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>

using namespace std;

void startCUDA(cv::cuda::GpuMat &src1, cv::cuda::GpuMat &src2,
               cv::cuda::GpuMat &dst, char operation, float scale_factor,
               float offset);

int main(int argc, char **argv) {
  cv::namedWindow("Original Image 1", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Original Image 2", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Processed Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);

  cv::Mat h_img1 = cv::imread(argv[1]);
  char operation = argv[2][0];
  cv::Mat h_img2 = cv::imread(argv[3]);
  float scale_factor = atof(argv[4]);
  float offset = atof(argv[5]);
  cv::Mat h_result(max(h_img1.rows, h_img2.rows), max(h_img1.cols, h_img2.cols),
                   CV_8UC3, cv::Scalar(0, 255, 0));
  cv::cuda::GpuMat d_img1, d_img2, d_result;

  cv::imshow("Original Image 1", h_img1);
  cv::imshow("Original Image 2", h_img2);

  auto begin = chrono::high_resolution_clock::now();
  const int iter = 1;

  d_img1.upload(h_img1);
  d_img2.upload(h_img1);
  d_result.upload(h_result);

  for (int i = 0; i < iter; i++) {
    startCUDA(d_img1, d_img2, d_result, operation, scale_factor, offset);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - begin;

  cv::imshow("Processed Image", d_result);

  cout << diff.count() << endl;
  cout << diff.count() / iter << endl;
  cout << iter / diff.count() << endl;

  cv::waitKey();
  return 0;

  return 0;
}
