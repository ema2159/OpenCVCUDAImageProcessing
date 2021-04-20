#include <chrono> // for high_resolution_clock
#include <iostream>
#include <math.h>
#include <omp.h>
#include <opencv2/opencv.hpp>

using namespace std;

/**
 * Shifts the hue of a pixel in the CIELAB colorspace.
 *
 * @param pix: pixel to transform, given in the CIELAB colorspace.
 * @param angle: angle to add to the pixel's hue angle.
 * @return hue shifter pixel.
 */
cv::Vec3f hueShift(cv::Vec3f pix, float angle) {
  float C_ab = sqrt(pow(pix[1], 2) + pow(pix[2], 2)); // sqrt((a*)^2+(b*)^2)
  float h = atan2(pix[2], pix[1]);

  // Shift CIELAB color's a* and b* components
  h += (angle * M_PI) / 180.0;

  return cv::Vec3f(pix[0], cos(h) * C_ab, sin(h) * C_ab);
}

int main(int argc, char **argv) {

  // Get images from command line
  cv::Mat source = cv::imread(argv[1], cv::IMREAD_COLOR);
  float hue_shift_angle = atof(argv[2]);

  cv::imshow("Source Image", source);

  // Placeholder image for the result
  cv::Mat destination(source.rows, source.cols, CV_8UC3, cv::Scalar(0, 255, 0));

  // Translate source image to the CIELAB colorspace
  cv::cvtColor(source, source, cv::COLOR_BGR2Lab);

  auto begin = chrono::high_resolution_clock::now();

#pragma omp parallel for
  for (int i = 0; i < source.rows; i++) {
    // #pragma omp parallel for
    for (int j = 0; j < source.cols; j++) {

      destination.at<cv::Vec3b>(i, j) =
          hueShift((cv::Vec3f)source.at<cv::Vec3b>(i, j), hue_shift_angle);
    }
  }

  // Translate result image to BGR colorspace
  cv::cvtColor(destination, destination, cv::COLOR_Lab2BGR);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - begin;

  cv::imshow("Processed Image", destination);

  cout << "Processing time: " << diff.count() << " s" << endl;

  cv::waitKey();
  return 0;
}
