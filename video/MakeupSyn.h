//#include <afx.h>
#include <iostream>
#include <string>
#include <vector>
#include "kptsDefine.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "CThinPlateSpline.h"
using namespace std;
#pragma once

struct Recom_Color
{
	float R;
	float G;
	float B;
};

#define SHAPENUM 20
class CMakeupSyn
{
public:
	CMakeupSyn(void);
	~CMakeupSyn(void);

	// the path of video to be processed
	std::string m_videoPath;
	// the path of face's key point of the video
	std::string m_kptPath;
	int m_recom_skin;
	int m_recom_lip;
	int m_recom_shad_color;
	int m_recom_shad_Id;
	void LoadMakeupFile(void);
	void InitMakeup(void);
	void InitMakeup_(void);
private:

	Recom_Color* m_temp_lip;
	Recom_Color* m_temp_skin;
	Recom_Color* m_temp_eyeshadow;
	cv::Mat m_temp_l_eyeshape;
	cv::Mat m_temp_r_eyeshape;
	 
	Recom_Color m_lip_color;
	Recom_Color m_skin_color;
	Recom_Color m_eyeshadow_color;

	cv::VideoCapture m_Vcapture;

	std::vector<cv::Point> m_kpts;
	std::vector<cv::Point> m_trainpts;
	std::vector<cv::Point> m_pts_l_eye;
	std::vector<cv::Point> m_pts_r_eye;
	
public:
	//int LoadVideoFile(CString filePath);
	int LoadVideoFile(char* filePath);

	bool LoadKptsFile(string filePath);
	bool LoadMeanShapeFile(string fileName);
	bool LoadMeanEyeShapeFile(string fileName);
	void MakeupFace(cv::Mat& Img, cv::Point2f * points);
	cv::Mat GetRoiloy(cv::Mat& srcImg, cv::Point2f* points, int* index, int flag, std::vector<cv::Point>& outPoint);
	cv::Mat CMakeupSyn::GetRoiloy(cv::Mat& srcImg, cv::Point2f* points, int* index, int flag);

	void ModifyEyeKpts(void);
	void ModifyEyeKpts_(void);
	void RGB2Lab(cv::Mat& src, cv::Mat& dst_It, cv::Mat& dst_Ic1, cv::Mat& dst_Ic2);
	void ProcessVideo(void);
	cv::Rect  GetFaceRoi(cv::Mat& src, cv::Mat& dst, std::vector<cv::Point>& kpts);
	cv::Mat GetRoiploy(cv::Size Imgsize,std::vector<cv::Point> kpts,int* index,int flag);
	void GetRoiloy(cv::Mat& srcImg, cv::Mat& dstImg, cv::Point2f* points, int* index, int flag);

	void GetRoiploy(cv::Mat& src, cv::Mat& dst,std::vector<cv::Point> kpts,int* index,int flag);
	
	cv::Mat GetRoiploy(cv::Size Imgsize,std::vector<cv::Point> kpts,
		int* index,int flag,std::vector<cv::Point>& cur_kpts);

	cv::Mat ConbineColor(cv::Mat& It_img, cv::Mat& Ic_img1,cv::Mat& Ic_img2, 
		cv::Mat& FaceImg,cv::Mat& skin_region,
		cv::Mat& lip_region,cv::Mat& eye_region);
private:
	void Change_data_color2(cv::Mat& Ic_img1, cv::Mat& Ic_img2, cv::Mat& It_img,
		cv::Mat& mask, float alpha, float alpha1, float alpha2,Recom_Color lab);
public:
	cv::Mat TPS_im_warp(const cv::Mat& TempImg, std::vector<cv::Point>& src_kpts,std::vector<cv::Point>& eye_kpts, int width, int height);
	cv::Mat Lab2RGB(cv::Mat& It_img, cv::Mat& Ic_img1, cv::Mat& Ic_img2);
	Recom_Color RGB2Lab(Recom_Color RGB);
	float Ft(float t);
	void MergeImage(cv::Mat& result, cv::Mat& src1, cv::Mat& smallsrc2,cv::Rect smallrect);
	void RGB2Lab(const cv::Mat& RGBImg, cv::Mat& It_img, cv::Mat& Ic_img1, cv::Mat& Ic_img2);
private:
	void InitTmat(void);
	cv::Mat Tmat;
	cv::Mat TTmat;

public:
	void  Lab2RGB(const cv::Mat& It_img, const cv::Mat& Ic_img1, const cv::Mat& Ic_img2,cv::Mat& dst);
};

