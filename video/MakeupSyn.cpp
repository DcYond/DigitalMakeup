#include "MakeupSyn.h"

//#include <mat.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <string>
using namespace cv;
using namespace std;

//#pragma comment(lib,"libmat.lib")
//#pragma comment(lib,"libmx.lib")
//#pragma comment(lib,"libmex.lib")
//#pragma comment(lib,"libeng.lib")



CMakeupSyn::CMakeupSyn(void)
	: m_videoPath("")
	, m_kptPath("")
	, m_recom_skin(0)
	, m_recom_lip(0)
	, m_recom_shad_color(0)
	, m_recom_shad_Id(0)
	, m_temp_lip(NULL)
	, m_temp_skin(NULL)
	, m_temp_eyeshadow(NULL)
{
	
	cv::Mat tmat(3,3,CV_32F);
	
	tmat.ptr<float>(0)[0] = 0.412453;
	tmat.ptr<float>(0)[1] = 0.357380;
	tmat.ptr<float>(0)[2] = 0.180423;

	tmat.ptr<float>(1)[0] = 0.212671;
	tmat.ptr<float>(1)[1] = 0.715160;
	tmat.ptr<float>(1)[2] = 0.072169;

	tmat.ptr<float>(2)[0] = 0.019334;
	tmat.ptr<float>(2)[1] = 0.119193;
	tmat.ptr<float>(2)[2] = 0.950227;

	Tmat = tmat;

	cv::Mat ttmat(3,3,CV_32F);

	tmat.ptr<float>(0)[0] = 3.240479;
	tmat.ptr<float>(0)[1] = -1.537150;
	tmat.ptr<float>(0)[2] = -0.498535;

	tmat.ptr<float>(1)[0] = -0.969256;
	tmat.ptr<float>(1)[1] = 1.875992;
	tmat.ptr<float>(1)[2] = 0.041556;

	tmat.ptr<float>(2)[0] = 0.055648;
	tmat.ptr<float>(2)[1] = -0.204043;
	tmat.ptr<float>(2)[2] = 1.057311;

	TTmat = ttmat;
}


CMakeupSyn::~CMakeupSyn(void)
{
}


void CMakeupSyn::LoadMakeupFile(void)
{
	
	std::ifstream inputFile("labcolor.txt", std::ios::in | std::ios::binary);
	if (!inputFile.is_open())
	{
		std::cout << "Error opening the file: makeup.txt\n";
		return;
	}
	std::cout << "loading makeup file...\n";
	m_temp_lip = new Recom_Color[15];
	m_temp_skin  = new Recom_Color[15];
	m_temp_eyeshadow = new Recom_Color[15];

	inputFile.read((char*)m_temp_lip, sizeof(Recom_Color) * 15);
	inputFile.read((char*)m_temp_skin, sizeof(Recom_Color) * 15);
	inputFile.read((char*)m_temp_eyeshadow, sizeof(Recom_Color) * 15);

	


	inputFile.close();


	
}

//void CMakeupSyn::LoadMakeupFile(void)
//{
//	const char *file;
//	file = "makeup_new.mat";
//
//	MATFile *pmatFile = NULL;
//	mxArray *pMxArray = NULL;
//	double* temp_idx;
//	int ndir;
//	const char **dir;
//
//	pmatFile = matOpen(file,"r");
//	if (pmatFile == NULL)
//	{
//		std::cout << "Error opening file:" << file << std::endl;
//		return;
//	}
//
//	dir = (const char**)matGetDir(pmatFile,&ndir);
//
//	if (dir == NULL)
//	{
//		std::cout << "Error reading directory of file:" << file << std::endl;
//		return;
//	}
//	else
//	{
//		std::cout << "mat Name:\n";
//		for (int i = 0; i < ndir; i++)
//		{
//			std::cout <<  dir[i] << std::endl;
//		}
//	}
//
//	
//	pMxArray = matGetVariable(pmatFile,"skin");
//	int M = mxGetM(pMxArray);
//	int N = mxGetN(pMxArray);		
//	temp_idx = (double*)mxGetData(pMxArray);
//
//	m_temp_skin = new double*[N];
//	for (int k = 0; k < N; k++)
//	{
//		m_temp_skin[k] = new double[M];
//		memcpy(m_temp_skin[k],(temp_idx+k*M),sizeof(double)*M);
//	}
//
//
//	pMxArray = matGetVariable(pmatFile,"lip");
//	M = mxGetM(pMxArray);
//	N = mxGetN(pMxArray);		
//	temp_idx = (double*)mxGetData(pMxArray);
//
//	m_temp_lip = new double*[N];
//	for (int k = 0; k < N; k++)
//	{
//		m_temp_lip[k] = new double[M];
//		memcpy(m_temp_lip[k],(temp_idx+k*M),sizeof(double)*M);
//	}
//
//	pMxArray = matGetVariable(pmatFile,"eyeshadow");
//	M = mxGetM(pMxArray);
//	N = mxGetN(pMxArray);		
//	temp_idx = (double*)mxGetData(pMxArray);
//
//	m_temp_eyeshadow = new double*[N];
//	for (int k = 0; k < N; k++)
//	{
//		m_temp_eyeshadow[k] = new double[M];
//		memcpy(m_temp_eyeshadow[k],(temp_idx+k*M),sizeof(double)*M);
//	}
//	
//}


void CMakeupSyn::InitMakeup(void)
{
	//for the skin color
	
		m_skin_color.R = m_temp_skin[m_recom_skin-1].R;
		m_skin_color.G = m_temp_skin[m_recom_skin-1].G;
		m_skin_color.B = m_temp_skin[m_recom_skin-1].B;
	
		m_lip_color.R = m_temp_lip[m_recom_lip-1].R /*+ 30*/;
		m_lip_color.G = m_temp_lip[m_recom_lip-1].G ;
		m_lip_color.B = m_temp_lip[m_recom_lip-1].B /*+ 110*/;

		m_eyeshadow_color.R = m_temp_eyeshadow[m_recom_shad_color-1].R;
		m_eyeshadow_color.G = m_temp_eyeshadow[m_recom_shad_color-1].G;
		m_eyeshadow_color.B = m_temp_eyeshadow[m_recom_shad_color-1].B;

		

		std::string Path = "eyeShape\\";
		std::stringstream ss;
		std::string str;
		ss << m_recom_shad_Id;
		ss >> str;
		Path += str;
		Path += ".jpg";
		//Path.AppendFormat("%d.jpg",m_recom_shad_Id);
		m_temp_l_eyeshape = cv::imread(Path,CV_LOAD_IMAGE_GRAYSCALE);
		cv::Mat temp;
		cv::flip(m_temp_l_eyeshape,temp,1);
		m_temp_r_eyeshape = temp.clone();
		

		
		LoadMeanShapeFile("meanAlignedShape_flip.txt");
		ModifyEyeKpts();
		
	
}

void CMakeupSyn::InitMakeup_(void)
{
	m_skin_color.R = m_temp_skin[m_recom_skin - 1].R;
	m_skin_color.G = m_temp_skin[m_recom_skin - 1].G;
	m_skin_color.B = m_temp_skin[m_recom_skin - 1].B;

	m_lip_color.R = m_temp_lip[m_recom_lip - 1].R /*+ 30*/;
	m_lip_color.G = m_temp_lip[m_recom_lip - 1].G;
	m_lip_color.B = m_temp_lip[m_recom_lip - 1].B /*+ 110*/;

	m_eyeshadow_color.R = m_temp_eyeshadow[m_recom_shad_color - 1].R;
	m_eyeshadow_color.G = m_temp_eyeshadow[m_recom_shad_color - 1].G;
	m_eyeshadow_color.B = m_temp_eyeshadow[m_recom_shad_color - 1].B;



	std::string Path = "eyeShape\\";
	std::stringstream ss;
	std::string str;
	ss << m_recom_shad_Id;
	ss >> str;
	Path += str;
	Path += ".jpg";
	//Path.AppendFormat("%d.jpg",m_recom_shad_Id);
	m_temp_l_eyeshape = cv::imread(Path, CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat temp;
	cv::flip(m_temp_l_eyeshape, temp, 1);
	m_temp_r_eyeshape = temp.clone();

	LoadMeanEyeShapeFile("MeanEyeShape.txt");
	ModifyEyeKpts_();
}

int CMakeupSyn::LoadVideoFile(char* filePath)
{
	//m_Vcapture.open(filePath.GetBuffer());
	m_Vcapture.open(filePath);
	if (!m_Vcapture.isOpened())
	{
		std::cout << "Error opening the video file: " << filePath << std::endl;
		return 0;
	}
	return 1;
}


bool CMakeupSyn::LoadKptsFile(string filePath)
{
	std::fstream inputFile;
	inputFile.open(filePath,std::ios::in);

	if (!inputFile.is_open())
	{
		std::cout << "Error fail to open file" << filePath.c_str() << std::endl;
		return false;
	}
	std::cout << "loading keypoints file...\n";
	int index = 0;
	int frame = 0;
	cv::Point p;
	//inputFile >> index;
	while (!inputFile.eof())
	{
		
		if (frame % 73 == 0)
		{
			inputFile >> index;
			
			
		}
		inputFile >> p.x >> p.y;
		m_kpts.push_back(p);

		frame++;
		
	}

	return true;
}


bool CMakeupSyn::LoadMeanShapeFile(string fileName)
{
	std::ifstream inputFile;
	inputFile.open(fileName,std::ios::in);

	if (!inputFile.is_open())
	{
		std::cout << "Error fail to open file" << fileName.c_str() << std::endl;
		return false;
	}
	cv::Point_<double> p;
	
	while(!inputFile.eof())
	{
		//inputFile >> x;
		inputFile >> p.x >> p.y;
		m_trainpts.push_back(p);
	}
	
	return true;
}

bool CMakeupSyn::LoadMeanEyeShapeFile(string fileName)
{
	std::ifstream inputFile;
	inputFile.open(fileName, std::ios::in);

	if (!inputFile.is_open())
	{
		std::cout << "Error fail to open file" << fileName.c_str() << std::endl;
		return false;
	}
	cv::Point_<double> p;

	while (!inputFile.eof())
	{
		//inputFile >> x;
		inputFile >> p.x >> p.y;
		m_trainpts.push_back(p);
	}

	return true;
}

void CMakeupSyn::ModifyEyeKpts(void)
{
	cv::Point p;
	std::vector<cv::Point> temp;
	for (int i = 0; i < EYE; i++)
	{
		p = m_trainpts.at(l_eye_s[i]-1);
		m_pts_l_eye.push_back(p);
		p.x = 513 - p.x;
		temp.push_back(p);
	}

	int modify[] = {5, 4, 3, 2, 1, 8, 7, 6};
	for (int k = 0; k < EYE; k++)
	{
		p = temp.at(modify[k]-1);

		m_pts_r_eye.push_back(p);
	}
}

void CMakeupSyn::ModifyEyeKpts_(void)
{
	cv::Point p;
	//std::vector<cv::Point> temp;
	for (int i = 0; i < EYE_; i++)
	{
		p = m_trainpts.at(i);
		m_pts_l_eye.push_back(p);
		p.x = 513 - p.x;
		//temp.push_back(p);
		m_pts_r_eye.push_back(p);
	}
}
void CMakeupSyn::RGB2Lab(cv::Mat& src, cv::Mat& dst_It, cv::Mat& dst_Ic1, cv::Mat& dst_Ic2)
{
	cv::Mat temImg;
	std::vector<cv::Mat> mv;
	

	/*cv::Mat floatImg;
	src.convertTo(floatImg,CV_32FC3);
	
	cv::cvtColor(floatImg,temImg,CV_RGB2Lab);*/


	cv::cvtColor(src,temImg,CV_RGB2Lab);
	cv::split(temImg,mv);

	//int d = mv[0].depth();

	/*uchar r = src.at<Vec3b>(0,0)[0];
	uchar g = src.at<Vec3b>(0,0)[1];
	uchar b = src.at<Vec3b>(0,0)[2];

	uchar l = temImg.at<Vec3b>(0,0)[0];
	uchar a = temImg.at<Vec3b>(0,0)[1];
	uchar B = temImg.at<Vec3b>(0,0)[2];*/
	

	//start = clock();
	dst_It = mv.at(0);
	dst_Ic1 = mv.at(1);
	dst_Ic2 = mv.at(2);


	/*std::vector<cv::Mat> twoMv;
	twoMv.push_back(mv.at(1));
	twoMv.push_back(mv.at(2));

	cv::merge(twoMv,dst_Ic);*/
	//stop = clock();
	//std::cout << "time for rgb2lab:"<< double (stop - start)/CLOCKS_PER_SEC << std::endl;
	
	

	//cv::imshow("Lab",temImg);
	/*cv::imshow("It",dst_It);
	cv::imshow("Ic",dst_Ic);*/
	//cv::waitKey(0);

}


void CMakeupSyn::ProcessVideo(void)
{
	std::cout << "process video...\n";
	int frameNum = m_Vcapture.get(CV_CAP_PROP_FRAME_COUNT);
	cv::Mat Img;
	cv::Mat FaceImg;
	std::vector<cv::Point> cur_kpts;
	m_Vcapture >> Img;

	cv::Size lastsize(Img.cols * 2,Img.rows);

	cv::Mat Result(lastsize,Img.type());
	clock_t start,finish;
	start = clock();
	for (int i = 0; i < frameNum-1; i++)
	{
		
		

		m_Vcapture >> Img;
		cur_kpts.assign(m_kpts.begin() + i*73, m_kpts.begin() + i*73+73);
		//start = clock();
		cv::Rect  rect = GetFaceRoi(Img,FaceImg,cur_kpts);
		//finish = clock();
		//std::cout <<"get faceroi: " << (double) (finish - start) / CLOCKS_PER_SEC << std::endl;

		std::vector<cv::Point> l_eye_kpts;
		std::vector<cv::Point> r_eye_kpts;

		//get the regions of face
		//start = clock();
		cv::Mat R_eye_region = GetRoiploy(FaceImg.size(),cur_kpts,r_eye,EYE,r_eye_kpts);
		cv::Mat L_eye_region = GetRoiploy(FaceImg.size(),cur_kpts,l_eye,EYE,l_eye_kpts);
		cv::Mat L_brow_region = GetRoiploy(FaceImg.size(),cur_kpts,l_brow,BROW);
		cv::Mat R_brow_region = GetRoiploy(FaceImg.size(),cur_kpts,r_brow,BROW);
		cv::Mat Mouth_region = 	GetRoiploy(FaceImg.size(),cur_kpts,mouth,MOUTH);
		cv::Mat Face_region = GetRoiploy(FaceImg.size(),cur_kpts,facial,FA_CE);
		//cv::Mat Face_region = GetRoiploy(FaceImg.size(),cur_kpts,facial,FACE);
		
		cv::Mat uplip_region = GetRoiploy(FaceImg.size(),cur_kpts,upper_lip,LIP);
		cv::Mat lowlip_region = GetRoiploy(FaceImg.size(),cur_kpts,lower_lip,LIP);

		uplip_region += lowlip_region;
		//finish = clock();
		//std::cout <<"get region: " << (double) (finish - start) / CLOCKS_PER_SEC << std::endl;


		//start = clock();
		cv::Mat TPS_L_eye = TPS_im_warp(m_temp_l_eyeshape,m_pts_l_eye,
			l_eye_kpts,FaceImg.cols,FaceImg.rows);
		cv::Mat TPS_R_eye = TPS_im_warp(m_temp_r_eyeshape,
			m_pts_r_eye,r_eye_kpts,FaceImg.cols,FaceImg.rows);
		TPS_R_eye += TPS_L_eye;
		//finish = clock();
		//std::cout <<"get TPS: " << (double) (finish - start) / CLOCKS_PER_SEC << std::endl;


		cv::Mat Skin_region = Face_region &(255-R_brow_region)
			& (255-L_brow_region) & (255-L_eye_region) & (255-R_eye_region)
			& (255-Mouth_region) & (255-TPS_R_eye);
		
		// transform RGB to Lab color space
		cv::Mat It_img,Ic_img1,Ic_img2;

		//start = clock();
		RGB2Lab(FaceImg,It_img,Ic_img1,Ic_img2);
		//finish = clock();
		//std::cout <<"rgb2lab: " << (double) (finish - start) / CLOCKS_PER_SEC << std::endl;


		
		cv::Mat mImg = ConbineColor(It_img,Ic_img1,Ic_img2,FaceImg,
			Skin_region,uplip_region,TPS_R_eye);
		
		MergeImage(Result,Img,mImg,rect);
		
		cv::imshow("Result",Result);
		cv::waitKey(1);

	}
	finish = clock();

	std::cout <<"average time for 1 image(ms): " << (double) (finish - start) / frameNum<< std::endl;

}

void CMakeupSyn::MakeupFace(cv::Mat& Img,cv::Point2f * points)
{
	std::vector<cv::Point> l_eye_kpts;
	std::vector<cv::Point> r_eye_kpts;

	cv::Mat L_eye_region = GetRoiloy(Img, points, l_eye_, EYE_, l_eye_kpts);
	cv::Mat R_eye_region = GetRoiloy(Img, points, r_eye_, EYE_,r_eye_kpts);
	
	//cv::Mat face_1 = GetRoiloy(Img, points, facial_, 5);

	cv::Mat Face_region = GetRoiloy(Img, points, facial_, FACE_);
	cv::Mat uplip_region = GetRoiloy(Img, points, upper_lip_, LIP_);
	cv::Mat lowlip_region = GetRoiloy(Img, points, lower_lip_, LIP_);
	uplip_region += lowlip_region;

	//start = clock();
	cv::Mat TPS_L_eye = TPS_im_warp(m_temp_l_eyeshape, m_pts_l_eye,
		l_eye_kpts, Img.cols, Img.rows);
	cv::Mat TPS_R_eye = TPS_im_warp(m_temp_r_eyeshape,
		m_pts_r_eye, r_eye_kpts, Img.cols, Img.rows);
	TPS_R_eye += TPS_L_eye;

	cv::Mat Skin_region = Face_region & (255 - L_eye_region) & (255 - R_eye_region)
		& (255 - uplip_region) & (255 - TPS_R_eye)/*| face_1*/;
	//cv::imshow("face",Skin_region);
	//cv::waitKey(0);
	// transform RGB to Lab color space
	cv::Mat It_img, Ic_img1, Ic_img2;

	//start = clock();
	RGB2Lab(Img, It_img, Ic_img1, Ic_img2);
	//finish = clock();
	//std::cout <<"rgb2lab: " << (double) (finish - start) / CLOCKS_PER_SEC << std::endl;



	cv::Mat mImg = ConbineColor(It_img, Ic_img1, Ic_img2, Img,
		Skin_region, uplip_region, TPS_R_eye);

	//MergeImage(Result, Img, mImg, rect);

	//cv::imshow("Result", mImg);
	Img = mImg;
	//cv::waitKey(1);

}
cv::Rect CMakeupSyn::GetFaceRoi(cv::Mat& src, cv::Mat& dst, std::vector<cv::Point>& kpts)
{
	int edge = 20;

	int min_x = 10000;
	int max_x = 0;
	int min_y = 10000;
	int max_y = 0;

	for (int i = 0; i < kpts.size(); i++)
	{
		if (kpts.at(i).x > max_x)
		{
			max_x = kpts.at(i).x;
		}

		if (kpts.at(i).x < min_x)
		{
			min_x = kpts.at(i).x;
		}

		if (kpts.at(i).y > max_y)
		{
			max_y = kpts.at(i).y;
		}

		if (kpts.at(i).y < min_y)
		{
			min_y = kpts.at(i).y;
		}

	}


	min_x = min_x - edge;
	max_x = max_x + edge;

	int height = kpts.at(39).y - kpts.at(32).y;

	min_y = min_y - height;
	max_y = max_y + edge;

	if (min_x < 0)
	{
		min_x = 0;
	}

	if (max_x > src.cols)
	{
		max_x = src.cols;
	}

	if (min_y < 0)
	{
		min_y = 0;
	}

	if (max_y > src.rows)
	{
		max_y = src.rows;
	}


	cv::Rect rect(min_x, min_y, max_x - min_x, max_y - min_y);
	dst = src(rect);

	for (int i = 0; i < kpts.size(); i++)
	{
		kpts.at(i).x = kpts.at(i).x - min_x;
		kpts.at(i).y = kpts.at(i).y - min_y;
	}

	return rect;
	
}


cv::Mat CMakeupSyn::GetRoiploy(cv::Size Imgsize,std::vector<cv::Point> kpts,int* index,int flag)
{
	std::vector<cv::Point> roikpts;
	cv::Point p;
	/*cv::Point **PP;
	PP = new cv::Point *[1];
	PP[0] = new cv::Point[EYE];*/
	cv::Point *PP = new cv::Point[flag];
	int npts[] = {flag};
	for (int i = 0; i < flag; i++)
	{
		p = kpts.at(index[i]-1);
		roikpts.push_back(p);
		PP[i] = p;
		//cv::circle(src,p,3,CV_RGB(0,255,0));
	}
	cv::Mat mask = cv::Mat::zeros(Imgsize,CV_8UC1);
	cv::fillConvexPoly(mask,PP,flag,CV_RGB(255,255,255));
	delete [] PP;
	PP = NULL;

	return mask;
}

cv::Mat CMakeupSyn::GetRoiploy(cv::Size Imgsize,std::vector<cv::Point> kpts,
	int* index,int flag,std::vector<cv::Point>& cur_kpts)
{
	
	cv::Point p;
	cv::Point *PP = new cv::Point[flag];
	int npts[] = {flag};
	for (int i = 0; i < flag; i++)
	{
		p = kpts.at(index[i]-1);
		cur_kpts.push_back(p);
		PP[i] = p;
		//cv::circle(src,p,3,CV_RGB(0,255,0));
	}
	cv::Mat mask = cv::Mat::zeros(Imgsize,CV_8UC1);
	cv::fillConvexPoly(mask,PP,flag,CV_RGB(255,255,255));
	delete [] PP;
	PP = NULL;
	return mask;
}


cv::Mat CMakeupSyn::ConbineColor(cv::Mat& It_img, cv::Mat& Ic_img1,cv::Mat& Ic_img2, cv::Mat& FaceImg,
	cv::Mat& skin_region,cv::Mat& lip_region,cv::Mat& eye_region)
{
	Change_data_color2(Ic_img1,Ic_img2,It_img,skin_region,0.15,0.8,0.2,m_skin_color);
	Change_data_color2(Ic_img1,Ic_img2,It_img,lip_region,0.3, 0.6, 0.4,m_lip_color);
	Change_data_color2(Ic_img1,Ic_img2,It_img,eye_region,0.6, 0.8, 0.2,m_eyeshadow_color);


	return Lab2RGB(It_img,Ic_img1,Ic_img2);
}


void CMakeupSyn::Change_data_color2(cv::Mat& Ic_img1, cv::Mat& Ic_img2, 
	cv::Mat& It_img, cv::Mat& mask,float alpha, float alpha1, 
	float alpha2,Recom_Color lab)
{
	float a = lab.R;
	float b = lab.G;
	float c = lab.B;

	/*cv::imshow("afIt",It_img);
	cv::Mat sk = It_img.mul(mask/255);
	cv::imshow("sk",sk);
	cv::Mat sm = Ic_img1.mul((255-mask)/255);
	cv::imshow("sm",sm);
	int type = mask.type();*/
	
	cv::Mat temp,temp2,temp3;

	/*cv::Mat Mask1 = mask / 255;
	cv::Mat Mask2 = 1-Mask1;

	cv::Mat sMask;
	cv::Mat rMask;

	Mask1.convertTo(sMask,CV_32F);
	Mask2.convertTo(rMask,CV_32F);
	int d = It_img.depth();
	d = sMask.depth();*/

	

	cv::Mat sMask = mask / 255;
	cv::Mat rMask = 1-sMask;
	//cv::Mat scaleMask = mask / 2;

	//cv::imshow("mask",mask);
	//cv::imshow("rmask",rMask);
	//cv::waitKey(0);
	
	cv::addWeighted(Ic_img1.mul(sMask),(1-alpha),sMask,alpha*b,0,temp);
	cv::add(Ic_img1.mul(rMask),temp,Ic_img1);
	//cv::imshow("temp",temp);

	cv::addWeighted(Ic_img2.mul(sMask),(1-alpha),sMask,alpha*c,0,temp2);
	cv::add(Ic_img2.mul(rMask),temp2,Ic_img2);
	//cv::imshow("temp2",temp2);

	cv::addWeighted(It_img.mul(sMask),alpha1,sMask,alpha2*a,0,temp3);
	cv::add(It_img.mul(rMask),temp3,It_img);

	/*cv::imshow("temp3",temp3);
	cv::imshow("afIt1",It_img);
	cv::imshow("afIc1",Ic_img1);
	cv::imshow("afIc2",Ic_img2);
	cv::waitKey(0);*/




	/*Ic_img1 = Ic_img1.mul(255-mask) + Ic_img1.mul(mask) * (1-alpha)
		+ mask * alpha1* LabMat.at<cv::Vec3f>(0,0)[1];

	Ic_img2 = Ic_img1.mul(255-mask) + Ic_img2.mul(mask) * (1-alpha)
		+ mask * alpha1* LabMat.at<cv::Vec3f>(0,0)[2];

	It_img = It_img.mul(255-mask) + It_img.mul(mask) * (alpha1)
		+ mask * alpha2* LabMat.at<cv::Vec3f>(0,0)[0];*/

}


cv::Mat CMakeupSyn::TPS_im_warp(const cv::Mat& TempImg, std::vector<cv::Point>& src_kpts,std::vector<cv::Point>& eye_kpts, int width, int height)
{

	 CThinPlateSpline tps(src_kpts,eye_kpts);
	 clock_t start,finish;
	// start = clock();
	 tps.computeSplineCoeffs(src_kpts,eye_kpts,0.01,BACK_WARP);
	 //finish = clock();
	// std::cout << "initial time " << double(finish - start)/ CLOCKS_PER_SEC << std::endl;
	 
	// start = clock();
	 Mat TPS = tps.pts(TempImg,width,height);	 
	// finish = clock();
	// std::cout << "tps time " << double(finish - start)/ CLOCKS_PER_SEC << std::endl;

	 
	 return TPS;	
}


cv::Mat CMakeupSyn::Lab2RGB(cv::Mat& It_img, cv::Mat& Ic_img1, cv::Mat& Ic_img2)
{
	cv::Mat mImg;
	cv::Mat RGBImg;
	cv::Mat mv[] = {It_img,Ic_img1,Ic_img2};
	cv::merge(mv,3,mImg);
	cv::cvtColor(mImg,RGBImg,CV_Lab2RGB);
	return RGBImg;
}

Recom_Color CMakeupSyn::RGB2Lab(Recom_Color RGB)
{
	int delta = 128;
	float R = RGB.R /255.0;
	float G = RGB.G /255.0;
	float B = RGB.B /255.0;
	
	float X = 0.412453 * R + 0.357580 * G + 0.180423 * B;
	float Y = 0.212671 * R + 0.715160 * G + 0.072169 * B;
	float Z = 0.019334 * R + 0.119193 * G + 0.950227 * B;

	X = X /(0.950456);
	
	Z = Z / (1.088754);


	Recom_Color result;
	if (Y > 0.008856)
	{ //L
		result.R = 116 * pow(float(Y),float(1.0/3.0)) - 16.0;
	} 
	else
	{
		result.R = 903.3 * Y;
	}
	//a
	result.G = 500 * (Ft(X) - Ft(Y)) + delta;
	//b
	result.B = 200 * (Ft(Y) - Ft(Z)) + delta;
	return result;
}




float CMakeupSyn::Ft(float t)
{
	if (t > 0.008856)
	{
		return pow(float(t),float(1.0/3.0));
	} 
	else
	{
		return 7.787 * t + 16.0 / 116.0;
	}
	
}


void CMakeupSyn::MergeImage(cv::Mat& result, cv::Mat& src1, cv::Mat& smallsrc2,cv::Rect smallrect)
{
	int rows = src1.rows;
	int cols = src1.cols;
	//将原图复制到左边
	cv::Rect rect1(0,0,cols,rows);
	cv::Mat leftMat = Mat(result,rect1);
	src1.copyTo(leftMat);
	//合成的图像复制到原图像中
	cv::Mat smallMat = Mat(src1,smallrect);
	smallsrc2.copyTo(smallMat);
	//将加入合成图像的原图像复制到右边
	cv::Rect rectr(cols,0,cols,rows);
	cv::Mat rightMat = Mat(result,rectr);
	src1.copyTo(rightMat);



}


void CMakeupSyn::RGB2Lab(const cv::Mat& RGBImg, cv::Mat& It_img, cv::Mat& Ic_img1, cv::Mat& Ic_img2)
{
	cv::Mat mv[3];
	cv::Mat Rgb = RGBImg / 255.0;
	cv::split(Rgb,mv);

	int T = 0.008856;
	

	int M = RGBImg.cols;
	int N = RGBImg.rows;

	int s = M * N;
	cv::Mat R;
	cv::Mat G;
	cv::Mat B;
	cv::resize(mv[0],R,cv::Size(1,s));
	cv::resize(mv[1],G,cv::Size(1,s));
	cv::resize(mv[2],B,cv::Size(1,s));

	cv::Mat reMv[] = {R,G,B};
	cv::Mat RGB;
	cv::merge(reMv,3,RGB);
	
	/*cv::Mat Tmat(3,3,CV_32F);
	
	Tmat.ptr<float>(0)[0] = 0.412453;
	Tmat.ptr<float>(0)[1] = 0.357380;
	Tmat.ptr<float>(0)[2] = 0.180423;

	Tmat.ptr<float>(1)[0] = 0.212671;
	Tmat.ptr<float>(1)[1] = 0.715160;
	Tmat.ptr<float>(1)[2] = 0.072169;

	Tmat.ptr<float>(2)[0] = 0.019334;
	Tmat.ptr<float>(2)[1] = 0.119193;
	Tmat.ptr<float>(2)[2] = 0.950227;*/

	cv::Mat XYZ = Tmat * RGB;
	cv::Mat rgbmv[3];
	cv::split(XYZ,rgbmv);

	rgbmv[0] = rgbmv[0] / 0.950456;
	rgbmv[2] = rgbmv[2] / 1.088754;

	cv::Mat XT = rgbmv[0] > T;
	cv::Mat YT = rgbmv[1] > T;
	cv::Mat ZT = rgbmv[1] > T;
	
	XT = XT - 255;
	YT = YT - 255;
	ZT = ZT - 255;

	cv::Mat powX;
	cv::Mat powY;
	cv::Mat powZ;

	cv::pow(rgbmv[0],(1.0/3.0),powX);
	cv::pow(rgbmv[1],(1.0/3.0),powY);
	cv::pow(rgbmv[2],(1.0/3.0),powZ);


	cv::Mat fX = XT.mul(powX) + (1-XT).mul(rgbmv[0].mul(7.87 + 16.0 / 116.0));
	cv::Mat fY = YT.mul(powY) + (1-YT).mul(rgbmv[1].mul(7.87 + 16.0 / 116.0));

	cv::Mat L = YT.mul(116.0 * powY - 16.0) + (1-YT).mul(rgbmv[1] * 903.3);

	cv::Mat fZ = ZT.mul(powZ) + (1-ZT).mul(rgbmv[2].mul(7.87 + 16.0 / 116.0));
	//cv::Mat RGB = cv::merge()

	cv::Mat a = 500 * (fX - fY);
	cv::Mat b = 200 * (fY - fZ);

	/*cv::Mat LL;
	cv::Mat aa;
	cv::Mat bb;*/

	cv::resize(L,It_img,cv::Size(M,N));
	cv::resize(a,Ic_img1,cv::Size(M,N));
	cv::resize(b,Ic_img2,cv::Size(M,N));

	/*cv::Mat labmv[] = {It_img,Ic_img1,Ic_img2};
	cv::Mat Lagimg;
	cv::merge(labmv,3,Lagimg);*/




}


void CMakeupSyn::InitTmat(void)
{
	
}


void CMakeupSyn::Lab2RGB(const cv::Mat& It_img,const cv::Mat& Ic_img1, 
	const cv::Mat& Ic_img2,cv::Mat& dst)
{
	int T1 = 0.008856;
	int T2 = 0.206893;
	int M = It_img.cols;
	int N = It_img.rows;

	int s = M * N;

	cv::Mat L;
	cv::Mat a;
	cv::Mat b;

	cv::resize(It_img,L,cv::Size(1,s));
	cv::resize(Ic_img1,a,cv::Size(1,s));
	cv::resize(Ic_img2,b,cv::Size(1,s));

	//compute Y
	cv::Mat fY;
	cv::pow(((L + 16.0) / 116),3,fY);
	cv::Mat YT = fY > T1;
	cv::Mat fYc = fY.clone();

	cv::Mat fY1;
	fY1 = (255-YT).mul(L / 903.3) + YT.mul(fY);

	cv::Mat Y = fY1;

	cv::Mat fY2;
	cv::Mat fYpow;
	cv::pow(fYc,(1.0/3.0),fYpow);

	fY2 = YT.mul(fYpow) + (255-YT).mul(fY1 * 7.787 + 16.0 / 116.0);

	//compute X
	cv::Mat fX = a / 500 + fY2;
	cv::Mat XT = fX > T2;
	cv::Mat fXpow;
	cv::pow(fX,3,fXpow);

	cv::Mat X = XT.mul(fXpow) + (255-XT).mul((fX - 16.0 / 116.0) / 7.787);

	//compute Z
	cv::Mat fZ = fY2 - b / 200;
	cv::Mat ZT = fZ > T2;
	cv::Mat fZpow;
	cv::pow(fZ,3,fZpow);

	cv::Mat Z = ZT.mul(fZpow) + (255-ZT).mul((fZ - 16.0/116.0)/7.787);

	X = X * 0.950456;
	Z = Z * 1.088754;

	cv::Mat XYZ(3,s,CV_32F);
	cv::Rect rect(0,0,s,1);
	cv::Mat Xr = Mat(XYZ,rect);
	X.copyTo(Xr);

	rect.x = 0,
	rect.y = 1,
	rect.width = s;
	rect.height = 1;
	cv::Mat Yr = Mat(XYZ,rect);
	Y.copyTo(Yr);

	rect.x = 0,
	rect.y = 2,
	rect.width = s;
	rect.height = 1;
	cv::Mat Zr = Mat(XYZ,rect);
	Z.copyTo(Zr);

	cv::Mat RGB = TTmat * XYZ;

	cv::Mat RGBs[3];
	cv::split(RGB,RGBs);

	cv::Mat R;
	cv::Mat G;
	cv::Mat B;
	cv::resize(RGBs[0],R,cv::Size(M,N));
	cv::resize(RGBs[1],G,cv::Size(M,N));
	cv::resize(RGBs[2],B,cv::Size(M,N));

	cv::Mat rgbmv[] = {R,G,B};
	//cv::Mat RGBimg;
	cv::merge(rgbmv,3,dst);



	//return RGBimg;
}


cv::Mat CMakeupSyn::GetRoiloy(cv::Mat& srcImg,cv::Point2f* points, int* index, int flag, std::vector<cv::Point>& outPoint)
{
	cv::Point * PP = new cv::Point[flag];
	cv::Point temp;
	for (int i = 0; i < flag; i++)
	{
		temp.x = PP[i].x = (int)(points[index[i]].x);
		temp.y = PP[i].y = (int)(points[index[i]].y);
		outPoint.push_back(temp);
		//cv::circle(srcImg, temp, 3, cv::Scalar(255, 0, 0));
		
 	}
	cv::Mat dstImg = cv::Mat::zeros(srcImg.size(), CV_8UC1);
	cv::fillConvexPoly(dstImg, PP, flag, CV_RGB(255, 255, 255));
	/*cv::imshow("src", srcImg);
	cv::imshow("img", dstImg);
	cv::waitKey(0);*/
	delete[] PP;
	PP = NULL;

	return dstImg;
}
cv::Mat CMakeupSyn::GetRoiloy(cv::Mat& srcImg, cv::Point2f* points, int* index, int flag)
{
	cv::Point * PP = new cv::Point[flag];
	//cv::Point temp;
	for (int i = 0; i < flag; i++)
	{
		PP[i].x = (int)(points[index[i]].x);
		PP[i].y = (int)(points[index[i]].y);
		//outPoint.push_back(temp);
		//cv::circle(srcImg, PP[i], 3, cv::Scalar(255, 0, 0));

	}
	cv::Mat dstImg = cv::Mat::zeros(srcImg.size(), CV_8UC1);
	cv::fillConvexPoly(dstImg, PP, flag, CV_RGB(255, 255, 255));
	/*cv::imshow("src", srcImg);*/
	/*cv::imshow("img", dstImg);
	cv::waitKey(0);*/
	delete[] PP;
	PP = NULL;

	return dstImg;
}