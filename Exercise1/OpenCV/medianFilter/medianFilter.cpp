#include <chrono> // for high_resolution_clock
#include <cstdlib>
#include <iostream>
#include <algorithm> // std::sort
#include <vector> // std::vector
#include <omp.h>
#include <opencv2/opencv.hpp>

using namespace std;

int main(int argc, char **argv) {

  cv::Mat source = cv::imread(argv[1], cv::IMREAD_COLOR);
  cv::Mat destination(source.rows, source.cols, CV_8UC3, cv::Scalar(0, 255, 0));

  cv::imshow("Source Image", source);

  auto begin = chrono::high_resolution_clock::now();

  const unsigned int KERNEL_SIZE = 10;
  const unsigned int KERNEL_POW_2 = pow(KERNEL_SIZE, 2);
  const unsigned int KERNEL_DIV_2 = KERNEL_SIZE / 2;

  // Create input image which corresponds to the source image with an added
  // replication padding.
  cv::Mat input;
  cv::copyMakeBorder(source, input, KERNEL_DIV_2, KERNEL_DIV_2, KERNEL_DIV_2,
                     KERNEL_DIV_2, cv::BORDER_REPLICATE);

#pragma omp parallel for
  // The KERNEL_DIV_2 start is to account for the added padding to the input
  // image.
  for (int i = KERNEL_DIV_2; i < source.rows + KERNEL_DIV_2; i++) {
    // #pragma omp parallel for
    for (int j = KERNEL_DIV_2; j < source.cols + KERNEL_DIV_2; j++) {
      std::vector<cv::Vec3f> av;
      for (int m = i - KERNEL_DIV_2; m <= i + KERNEL_DIV_2; m++) {
        for (int n = j - KERNEL_DIV_2; n <= j + KERNEL_DIV_2; n++) {
	  av.push_back(input.at<cv::Vec3b>(m, n));
        }
      }
      std::nth_element(
	  av.begin(), av.begin() + KERNEL_POW_2/2, av.end(),
	  [](cv::Vec3b pix1, cv::Vec3b pix2) {
	    float length1 = pow(pix1[0], 2) + pow(pix1[1], 2) + pow(pix1[2], 2);
	    float length2 = pow(pix2[0], 2) + pow(pix2[1], 2) + pow(pix2[2], 2);
	    return length1 > length2;
	  });
      cv::Vec3f median;
      median = av[KERNEL_POW_2/2];

      destination.at<cv::Vec3b>(i - KERNEL_DIV_2, j - KERNEL_DIV_2) = median;
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - begin;

  cv::imshow("Processed Image", destination);

  cout << "Processing time: " << diff.count() << " s" << endl;

  cv::waitKey();
  return 0;
}
