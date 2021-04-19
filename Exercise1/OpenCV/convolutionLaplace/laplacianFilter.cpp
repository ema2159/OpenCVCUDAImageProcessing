#include <chrono> // for high_resolution_clock
#include <iostream>
#include <omp.h>
#include <opencv2/opencv.hpp>

using namespace std;

int main(int argc, char **argv) {

  cv::Mat source = cv::imread(argv[1], cv::IMREAD_COLOR);
  cv::Mat destination(source.rows, source.cols, CV_8UC3, cv::Scalar(0, 255, 0));

  cv::imshow("Source Image", source);

  cout << "Threads: " << omp_get_num_threads() << endl;

  auto begin = chrono::high_resolution_clock::now();
  const int iter = 1;

  const unsigned int KERNEL_SIZE = 3;
  const unsigned int KERNEL_DIV_2 = KERNEL_SIZE / 2;

  // clang-format off
  const int laplace_mat[3][3] = {
    {-1, -1, -1},
    {-1,  8, -1},
    {-1, -1, -1}
  };
  // clang-format on

  // Create input image which corresponds to the source image with an added
  // replication padding.
  cv::Mat input;
  cv::copyMakeBorder(source, input, KERNEL_DIV_2, KERNEL_DIV_2, KERNEL_DIV_2,
                     KERNEL_DIV_2, cv::BORDER_REPLICATE);

  for (int it = 0; it < iter; it++) {
#pragma omp parallel for
    // The KERNEL_DIV_2 start is to accout for the added padding to the input
    // image.
    for (int i = KERNEL_DIV_2; i < source.rows + KERNEL_DIV_2; i++) {
      // #pragma omp parallel for
      for (int j = KERNEL_DIV_2; j < source.cols + KERNEL_DIV_2; j++) {
        cv::Vec3f av = cv::Vec3f(0, 0, 0);
        for (int m = i - (KERNEL_SIZE / 2); m <= i + (KERNEL_SIZE / 2); m++) {
          for (int n = j - (KERNEL_SIZE / 2); n <= j + (KERNEL_SIZE / 2); n++) {
            av += (cv::Vec3f)input.at<cv::Vec3b>(m, n) *
                  laplace_mat[m - i + KERNEL_DIV_2][n - j + KERNEL_DIV_2];
          }
        }
        destination.at<cv::Vec3b>(i - KERNEL_DIV_2, j - KERNEL_DIV_2) = av;
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
