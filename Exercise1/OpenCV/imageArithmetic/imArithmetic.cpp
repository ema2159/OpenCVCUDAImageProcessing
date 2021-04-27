#include <chrono> // for high_resolution_clock
#include <iostream>
#include <omp.h>
#include <opencv2/opencv.hpp>

using namespace std;

/**
 * Clamps a vector's values between low and high.
 *
 * @param elem: element to clamp.
 * @param low: lower bound for the clamp.
 * @param high: upper bound for the clamp.
 * @return a vector with the original vector values clamped in the specified
 * interval.
 */
cv::Vec3f clamp(cv::Vec3f elem, float low, float high) {
  cv::Vec3f result;
  result[0] = max(low, min(high, elem[0]));
  result[1] = max(low, min(high, elem[1]));
  result[2] = max(low, min(high, elem[2]));

  return result;
}

/**
 * Do an arithmetic operation between two pixels.
 * I does the following opertion:
 *
 * (pix1 (operation) pix2) * scale_factor + offset
 *
 * @param pix1: corresponds to the first pixel argument.
 * @param pix2: corresponds to the second pixel argument.
 * @param operation: the operator to utilize. Choose between +, -, * and /
 * @param scale_factor: scale factor for the pixel operation.
 * @param offest: offset for the pixel operation.
 * @return pixel that results from the operation.
 */
cv::Vec3f pix_arithm(cv::Vec3f pix1, cv::Vec3f pix2, char operation,
                     float scale_factor, float offset) {
  cv::Vec3f result;
  cv::Vec3f vec_offset = cv::Vec3f(offset);
  switch (operation) {
  case '+': {
    result = pix1 + pix2;
    break;
  }
  case '-': {
    result = pix1 - pix2;
    break;
  }
  case 'x': {
    result[0] = pix1[0] * pix2[0];
    result[1] = pix1[1] * pix2[1];
    result[2] = pix1[2] * pix2[2];
    break;
  }
  case '/': {
    result[0] = pix1[0] / pix2[0];
    result[1] = pix1[1] / pix2[1];
    result[2] = pix1[2] / pix2[2];
    break;
  }
  default:
    cout << "INVALID OPERATOR" << endl;
    result = cv::Vec3f(0, 0, 0);
    break;
  }
  result = result * scale_factor + vec_offset;
  return result;
}

int main(int argc, char **argv) {

  // Get images from command line
  cv::Mat source1 = cv::imread(argv[1], cv::IMREAD_COLOR);
  cv::Mat source2 = cv::imread(argv[3], cv::IMREAD_COLOR);

  // Get operation parameters from the command line
  char operation = argv[2][0];
  float sc_factor = atof(argv[4]);
  float offset = atof(argv[5]);

  cv::imshow("Source Image 1", source1);
  cv::imshow("Source Image 2", source2);

  // Placeholder image for the result
  cv::Mat destination(source1.rows, source1.cols, CV_8UC3,
                      cv::Scalar(0, 255, 0));

  auto begin = chrono::high_resolution_clock::now();

  const int iter = 10;
  for (int it = 0; it < iter; it++) {
#pragma omp parallel for
    for (int i = 0; i < source1.rows; i++) {
      // #pragma omp parallel for
      for (int j = 0; j < source1.cols; j++) {
        cv::Vec3f av;
        av = pix_arithm((cv::Vec3f)source1.at<cv::Vec3b>(i, j),
                        (cv::Vec3f)source2.at<cv::Vec3b>(i, j), operation,
                        sc_factor, offset);

        av = clamp(av, 0, 255);

        destination.at<cv::Vec3b>(i, j) = av;
      }
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - begin;

  cv::imshow("Processed Image", destination);

  cout << "Processing time: " << diff.count() << " s" << endl;
  cout << "Time for 1 iteration: " << diff.count() / iter << " s" << endl;
  cout << "IPS: " << iter / diff.count() << endl;

  cv::waitKey();
  return 0;
}
