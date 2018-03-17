

#ifndef __PAW_h_
#define __PAW_h_


#include <opencv2/core/core.hpp>

namespace KeyPointsDetector
{
 

class PAW
{
public:    
	// Number of pixels after the warping to neutral shape
    int     number_of_pixels; 

	// Minimum x coordinate in destination
    double  min_x;

	// minimum y coordinate in destination
    double  min_y;

	// Destination points (landmarks to be warped to)
	cv::Mat_<double> destination_landmarks;

	// Destination points (landmarks to be warped from)
	cv::Mat_<double> source_landmarks;

	// Triangulation, each triangle is warped using an affine transform
	cv::Mat_<int> triangulation;

	// Triangle index, indicating which triangle each of destination pixels lies in
	cv::Mat_<int> triangle_id;

	// Indicating if the destination warped pixels is valid (lies within a face)
	cv::Mat_<uchar> pixel_mask;

	// A number of precomputed coefficients that are helpful for quick warping
	
	
	cv::Mat_<double> coefficients;

	// matrix of (c,x,y) coeffs for alpha
	cv::Mat_<double> alpha;

	// matrix of (c,x,y) coeffs for alpha
	cv::Mat_<double> beta;

	// x-source of warped points
	cv::Mat_<float> map_x;

	// y-source of warped points
	cv::Mat_<float> map_y;

	// Default constructor
    PAW(){;}

	// Construct a warp from a destination shape and triangulation
	PAW(const cv::Mat_<double>& destination_shape, const cv::Mat_<int>& triangulation);

	PAW(const cv::Mat_<double>& destination_shape, const cv::Mat_<int>& triangulation, double in_min_x, double in_min_y, double in_max_x, double in_max_y);

	// Copy constructor
	PAW(const PAW& other);

	void Read(std::ifstream &s);

	// The actual warping
    void Warp(const cv::Mat& image_to_warp, cv::Mat& destination_image, const cv::Mat_<double>& landmarks_to_warp);
	
	// Compute coefficients needed for warping
    void CalcCoeff();

	// Perform the actual warping
    void WarpRegion(cv::Mat_<float>& map_x, cv::Mat_<float>& map_y);

    inline int NumberOfLandmarks() const {return destination_landmarks.rows/2;} ;
    inline int NumberOfTriangles() const {return triangulation.rows;} ;

	// The width and height of the warped image
    inline int constWidth() const {return pixel_mask.cols;}
    inline int Height() const {return pixel_mask.rows;}
    
private:

	int findTriangle(const cv::Point_<double>& point, const std::vector<std::vector<double>>& control_points, int guess = -1) const;

  };
  //===========================================================================
}
#endif
