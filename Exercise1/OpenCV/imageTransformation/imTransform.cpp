#include <chrono> // for high_resolution_clock
#include <iostream>
#include <math.h> /* modf */
#include <omp.h>
#include <opencv2/opencv.hpp>

using namespace std;

/**
 * Returns the linear interpolation of a and b by a factor of t.
 * Performs the following operation:
 * result = a + t*(b - a)
 *
 * @param a: first value to interpolate.
 * @param b: second value to interpolate.
 * @param t: interpolation factor.
 * @return result of the interpolation.
 */
float lerp(float a, float b, float t) { return a + t * (b - a); }

/**
 * Returns the bilinear interpolation of four pixels.
 * Performs the following operation:
 * inpH1 = lerp(pix1, pix2, intpX)
 * inpH2 = lerp(pix3, pix4, intpX)
 * inpV = lerp(inpH1, inpH2, intpY)
 *
 * @param pix1: first pixel to interpolate.
 * @param pix2: second pixel to interpolate.
 * @param pix3: third pixel to interpolate.
 * @param pix4: fourth pixel to interpolate.
 * @param intpX: horizontal interpolation factor.
 * @param intpY: vertical interpolation factor.
 * @return result of the bilinear interpolation.
 */
cv::Vec3f interpolate_pix(cv::Vec3f pix1, cv::Vec3f pix2, cv::Vec3f pix3,
                          cv::Vec3f pix4, float intpX, float intpY) {
  // First horizontal interpolation
  float B_h1 = lerp(pix1[0], pix2[0], intpX);
  float G_h1 = lerp(pix1[1], pix2[1], intpX);
  float R_h1 = lerp(pix1[2], pix2[2], intpX);
  // Second horizontal interpolation
  float B_h2 = lerp(pix3[0], pix4[0], intpX);
  float G_h2 = lerp(pix3[1], pix4[1], intpX);
  float R_h2 = lerp(pix3[2], pix4[2], intpX);
  // Vertical interpolation
  float B = lerp(B_h1, B_h2, intpY);
  float G = lerp(G_h1, G_h2, intpY);
  float R = lerp(R_h1, R_h2, intpY);

  return cv::Vec3f(B, G, R);
}

int main(int argc, char **argv) {

  // Get images from command line
  cv::Mat source = cv::imread(argv[1], cv::IMREAD_COLOR);

  // Get operation parameters from the command line
  float scaleX = min(atof(argv[2]), 5.0);
  float scaleY = min(atof(argv[3]), 5.0);

  cv::imshow("Source Image 1", source);

  // Placeholder image for the result
  cv::Mat destination(source.rows * scaleX, source.cols * scaleY, CV_8UC3,
                      cv::Scalar(0, 255, 0));

  auto begin = chrono::high_resolution_clock::now();

  const int iter = 10;
  for (int it = 0; it < iter; it++) {
#pragma omp parallel for
    for (int i = 0; i < destination.rows; i++) {
      // #pragma omp parallel for
      for (int j = 0; j < destination.cols; j++) {
        float x_pos, y_pos, intpX, intpY;
        intpX = modf(i / scaleX, &x_pos);
        intpY = modf(j / scaleY, &y_pos);
        cv::Vec3f pix1 = (cv::Vec3f)source.at<cv::Vec3b>(x_pos, y_pos);
        cv::Vec3f pix2 = (cv::Vec3f)source.at<cv::Vec3b>(x_pos + 1, y_pos);
        cv::Vec3f pix3 = (cv::Vec3f)source.at<cv::Vec3b>(x_pos, y_pos + 1);
        cv::Vec3f pix4 = (cv::Vec3f)source.at<cv::Vec3b>(x_pos + 1, y_pos + 1);
        destination.at<cv::Vec3b>(i, j) =
            interpolate_pix(pix1, pix2, pix3, pix4, intpX, intpY);
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
