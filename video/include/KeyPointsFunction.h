



#ifndef __KEY_POINTS_DETECTOR_FUNCTION_h_
#define __KEY_POINTS_DETECTOR_FUNCTION_h_

// OpenCV includes
#include <opencv2/core/core.hpp>

#include <KeyPointsParameters.h>
//#include <KeyPointsDetectorUtils.h>
#include <KeyPointsCore.h>

using namespace std;

namespace KeyPointsDetector
{

	
	bool DetectKeyPointsInVideo(const cv::Mat_<uchar> &grayscale_image, FACE& FACE_model, FaceModelParameters& params);
	bool DetectKeyPointsInVideo(const cv::Mat_<uchar> &grayscale_image, const cv::Mat_<float> &depth_image, FACE& FACE_model, FaceModelParameters& params);

	bool DetectKeyPointsInVideo(const cv::Mat_<uchar> &grayscale_image, const cv::Rect_<double> bounding_box, FACE& FACE_model, FaceModelParameters& params);
	bool DetectKeyPointsInVideo(const cv::Mat_<uchar> &grayscale_image, const cv::Mat_<float> &depth_image, const cv::Rect_<double> bounding_box, FACE& FACE_model, FaceModelParameters& params);

	bool DetectKeyPointsInImage(const cv::Mat_<uchar> &grayscale_image, FACE& FACE_model, FaceModelParameters& params);
	// Providing a bounding box
	bool DetectKeyPointsInImage(const cv::Mat_<uchar> &grayscale_image, const cv::Rect_<double> bounding_box, FACE& FACE_model, FaceModelParameters& params);

	bool DetectKeyPointsInImage(const cv::Mat_<uchar> &grayscale_image, const cv::Mat_<float> depth_image, FACE& FACE_model, FaceModelParameters& params);
	bool DetectKeyPointsInImage(const cv::Mat_<uchar> &grayscale_image, const cv::Mat_<float> depth_image, const cv::Rect_<double> bounding_box, FACE& FACE_model, FaceModelParameters& params);
	
	void matchTemplate_m(const cv::Mat_<float>& input_img, cv::Mat_<double>& img_dft, cv::Mat& _integral_img, cv::Mat& _integral_img_sq, const cv::Mat_<float>&  templ, map<int, cv::Mat_<double> >& templ_dfts, cv::Mat_<float>& result, int method);



	//This assumes that align_from and align_to are already mean normalised
	cv::Matx22d AlignShapesKabsch2D(const cv::Mat_<double>& align_from, const cv::Mat_<double>& align_to);


	// Basically Kabsch's algorithm but also allows the collection of points to be different in scale from each other
	cv::Matx22d AlignShapesWithScale(cv::Mat_<double>& src, cv::Mat_<double> dst);


	void Project(cv::Mat_<double>& dest, const cv::Mat_<double>& mesh, double fx, double fy, double cx, double cy);
	void DrawBox(cv::Mat image, cv::Vec6d pose, cv::Scalar color, int thickness, float fx, float fy, float cx, float cy);

	// Drawing face bounding box
	vector<pair<cv::Point, cv::Point>> CalculateBox(cv::Vec6d pose, float fx, float fy, float cx, float cy);
	void DrawBox(vector<pair<cv::Point, cv::Point>> lines, cv::Mat image, cv::Scalar color, int thickness);

	vector<cv::Point2d> CalculateLandmarks(const cv::Mat_<double>& shape2D, cv::Mat_<int>& visibilities);
	vector<cv::Point2d> CalculateLandmarks(FACE& FACE_model);
	void DrawLandmarks(cv::Mat img, vector<cv::Point> landmarks);

	void Draw(cv::Mat img, const cv::Mat_<double>& shape2D, const cv::Mat_<int>& visibilities);
	void Draw(cv::Mat img, const cv::Mat_<double>& shape2D);
	void Draw(cv::Mat img, const FACE& FACE_model);



	cv::Matx33d Euler2RotationMatrix(const cv::Vec3d& eulerAngles);

	// Using the XYZ convention R = Rx * Ry * Rz, left-handed positive sign
	cv::Vec3d RotationMatrix2Euler(const cv::Matx33d& rotation_matrix);

	cv::Vec3d AxisAngle2Euler(const cv::Vec3d& axis_angle);

	cv::Vec3d RotationMatrix2AxisAngle(const cv::Matx33d& rotation_matrix);


	// Face detection using Haar cascade classifier
	bool DetectFaces(vector<cv::Rect_<double> >& o_regions, const cv::Mat_<uchar>& intensity);
	bool DetectFaces(vector<cv::Rect_<double> >& o_regions, const cv::Mat_<uchar>& intensity, cv::CascadeClassifier& classifier);
	// The preference point allows for disambiguation if multiple faces are present (pick the closest one), if it is not set the biggest face is chosen
	bool DetectSingleFace(cv::Rect_<double>& o_region, const cv::Mat_<uchar>& intensity, cv::CascadeClassifier& classifier, const cv::Point preference = cv::Point(-1, -1));


	// Reading a matrix written in a binary format
	void ReadMatBin(std::ifstream& stream, cv::Mat &output_mat);

	// Reading in a matrix from a stream
	void ReadMat(std::ifstream& stream, cv::Mat& output_matrix);

	// Skipping comments (lines starting with # symbol)
	void SkipComments(std::ifstream& stream);
	
}
#endif
