#include "KeyPoints.h"
#include "MakeupSyn.h"
#include <fstream>
#include <sstream>
#include <iostream>
// OpenCV includes
#include<opencv2/opencv.hpp>
const int PointsNum = 3;

void Get17Keypoints(cv::Mat_<double> FaceShape, cv::Mat_<int>visibilities, cv::Point2f* OutPoints)
{
	int n = FaceShape.rows / 2;
	for (int i = 0; i < 17; i++)
	{
		if (visibilities.at<int>(i))
		{

			OutPoints[i].x = FaceShape.at<double>(i);
			OutPoints[i].y = FaceShape.at<double>(i + n);
		}
	}

}
void DrawImg(cv::Mat Img, KeyPointsDetector::FACE& model, cv::Point2f* points)
{
	int idx = model.patch.GetViewIdx(model.params_global, 0);
	int n = model.detected_keypoints.rows / 2;
	cv::Mat_<double> shape2D = model.detected_keypoints;
	cv::Mat_<int> visibilities = model.patch.visibilities[0][idx];
	for (int i = 0; i < n; i++)
	{
		if (visibilities.at<int>(i))
		{
			cv::Point featurePoint((int)shape2D.at<double>(i), (int)shape2D.at<double>(i + n));

			if (i < PointsNum)
			{
				points[i].x = shape2D.at<double>(i);
				points[i].y = shape2D.at<double>(i + n);

			}
			// A rough heuristic for drawn point size
			int thickness = (int)std::ceil(3.0* ((double)Img.cols) / 640.0);
			int thickness_2 = (int)std::ceil(1.0* ((double)Img.cols) / 640.0);

			cv::circle(Img, featurePoint, 1, cv::Scalar(0, 0, 255), thickness);
			/*char fpsC[255];
			std::sprintf(fpsC, "%d", (int)i + 1);
			string fpsSt = fpsC;

			cv::putText(Img, fpsSt, featurePoint, CV_FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(255, 0, 0));*/
			//cv::circle(img, featurePoint, 1, cv::Scalar(255,0,0), thickness_2);
		}
	}
}
cv::Rect GetRoiFace(cv::Mat_<double>currentShape, int Width, int Height, int NumberOfPoints)
{

	double min_x;
	double max_x;
	cv::minMaxLoc(currentShape(cv::Rect(0, 0, 1, NumberOfPoints)), &min_x, &max_x);

	double min_y;
	double max_y;
	cv::minMaxLoc(currentShape(cv::Rect(0, NumberOfPoints, 1, NumberOfPoints)), &min_y, &max_y);

	double width = abs(min_x - max_x);
	double height = abs(min_y - max_y);


	double stepW = width / 2;
	double stepH = height / 2;

	if (min_x - stepW >= 0)
	{
		min_x -= stepW;
	}
	else
	{
		min_x = 0.0;
	}

	if (min_y - stepH >= 0)
	{
		min_y -= stepH;
	}
	else
	{
		min_y = 0.0;
	}

	if (max_x + stepW <= Width)
	{
		max_x += stepW;
	}
	else
	{
		max_x = Width;
	}

	if (max_y + stepH <= Height)
	{
		max_y += stepH;
	}
	else
	{
		max_y = Height;
	}

	

	width = abs(min_x - max_x);
	height = abs(min_y - max_y);
	return cv::Rect((int)min_x, (int)min_y, (int)width, (int)height);
}
void  GetRoiKeyPoints(cv::Rect Roi, cv::Mat_<double>currentShape, int NumberOfPoints, cv::Point2f* points)
{
	cv::Mat_<double> ResultShape = cv::Mat::zeros(currentShape.rows, currentShape.cols, currentShape.type());
	cv::Mat_<double> x = currentShape(cv::Rect(0, 0, 1, NumberOfPoints));
	x = x - Roi.x;
	cv::Mat_<double> y = currentShape(cv::Rect(0, NumberOfPoints, 1, NumberOfPoints));
	y = y - Roi.y;
	for (int i = 0; i < PointsNum; i++)
	{
		points[i].x = x.at<double>(i);
		points[i].y = x.at<double>(i);

	}

}

void GetTranformPoints(cv::Mat_<double> shape2D, cv::Point2f* points)
{

	points[0].x = shape2D.at<double>(0);
	points[0].y = shape2D.at<double>(0 + 68);
	points[1].x = shape2D.at<double>(8);
	points[1].y = shape2D.at<double>(8 + 68);
	points[2].x = shape2D.at<double>(16);
	points[2].y = shape2D.at<double>(16 + 68);


}
void HairSynthesis(cv::Mat& dstImg, cv::Point2f* dstPoints, cv::Mat& srcImg, cv::Point2f* srcPoints, cv::Mat& srcMask, cv::Mat& outResult)
{
	cv::Mat WarpMat;
	cv::Mat ImgDst;
	cv::Mat MaskDst;

	WarpMat = cv::getAffineTransform(srcPoints, dstPoints);
	cv::warpAffine(srcImg, ImgDst, WarpMat, dstImg.size());
	cv::warpAffine(srcMask, MaskDst, WarpMat, dstImg.size());
	outResult = ImgDst.mul(MaskDst) + dstImg.mul(cv::Scalar(1, 1, 1) - MaskDst);

}
void GetKeyPoints(cv::Mat_<double> shape2D, cv::Point2f* points)
{
	for (int i = 0; i < 68; i++)
	{		
		points[i].x = shape2D.at<double>(i);
		points[i].y = shape2D.at<double>(i + 68);
		
	}
}



int main(int argc,char** argv)
{
	KeyPointsDetector::FaceModelParameters det_parameters;
	KeyPointsDetector::FACE detect_model(det_parameters.model_location);
	
	cv::VideoCapture video_capture;
	video_capture = cv::VideoCapture("zt_640.avi");
	if (!video_capture.isOpened())
		return -1;
	cv::Mat Img;
	cv::Mat gray;
	video_capture >> Img;
	//cv::flip(Img, Img, 1);
	
	bool isHair = true;
	cv::Mat HairImg;
	cv::Mat HairGray;
	cv::Mat HairMask;

	cv::Mat warp_mat;
	cv::Mat warp_dst;
	cv::Mat hair_dst;
	cv::Mat Result;
	warp_dst = cv::Mat::zeros(Img.rows, Img.cols, Img.type());
	hair_dst = cv::Mat::zeros(Img.rows, Img.cols, Img.type());

	cv::Point2f hairPoints[PointsNum];
	cv::Point2f DstPoints[PointsNum];
	cv::Point2f FacePoints[68];
	
	CMakeupSyn FaceMakeup;

	if (isHair)
	{
		HairImg = cv::imread("Anne.jpg");
		HairMask = cv::imread("Anne.png");//0-1图
		cv::Mat HairGray;
		cv::cvtColor(HairImg, HairGray, CV_BGR2GRAY);
		KeyPointsDetector::DetectKeyPointsInImage(HairGray, detect_model, det_parameters);
	
		GetTranformPoints(detect_model.detected_keypoints, hairPoints);
		//cv::imshow("Hair", HairImg);

		FaceMakeup.m_recom_skin = 6;
		FaceMakeup.m_recom_lip = 2;
		FaceMakeup.m_recom_shad_color = 2;
		FaceMakeup.m_recom_shad_Id = 1;
		FaceMakeup.LoadMakeupFile();
		FaceMakeup.InitMakeup_();
		//cv::waitKey(0);

	}
	cv::Mat flipImg;
	while (!Img.empty())
	{
		cv::imshow("srcImg", Img);
		cv::cvtColor(Img, gray, CV_BGR2GRAY);
		bool detection_success = KeyPointsDetector::DetectKeyPointsInVideo(gray, detect_model, det_parameters);

		if (isHair && detection_success)
		{
			GetKeyPoints(detect_model.detected_keypoints, FacePoints);
			//上底妆
			FaceMakeup.MakeupFace(Img, FacePoints);

			GetTranformPoints(detect_model.detected_keypoints, DstPoints);
			//发型仿射变换
			HairSynthesis(Img, DstPoints, HairImg, hairPoints,HairMask, Result);
			cv::imshow("vid", Result);
			
		}
		else
		{
			KeyPointsDetector::Draw(Img, detect_model);
			cv::imshow("vid", Img);
		}
		
		cv::waitKey(1);
		video_capture >> Img;

	}
	return 1;
}