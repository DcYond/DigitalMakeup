/*
* CThinPlateSpline.cpp
*
*  Created on: 24.01.2010
*      Author: schmiedm
*/

#include <vector>
//#include <cv.h>
//#include <cxcore.h>
#include <opencv2\opencv.hpp>
#include "CThinPlateSpline.h"
using namespace cv;
#include <fstream>

CThinPlateSpline::CThinPlateSpline() {
        FLAG_MAPS_FORWARD_SET = false;
        FLAG_MAPS_BACK_WARP_SET = false;
        FLAG_COEFFS_BACK_WARP_SET = false;
        FLAG_COEFFS_FORWARD_WARP_SET = false;
}

CThinPlateSpline::CThinPlateSpline(const std::vector<Point>& pS, const std::vector<Point>& pD)
{
        if(pS.size() == pS.size())
        {
                pSrc = pS;
                pDst = pD;
        }

        FLAG_MAPS_FORWARD_SET = false;
        FLAG_MAPS_BACK_WARP_SET = false;
        FLAG_COEFFS_BACK_WARP_SET = false;
        FLAG_COEFFS_FORWARD_WARP_SET = false;
}

CThinPlateSpline::~CThinPlateSpline() {
}

void CThinPlateSpline::addCorrespondence(const Point& pS, const Point& pD)
{
        pSrc.push_back(pS);
        pDst.push_back(pD);

        // tell the class to recompute the coefficients if neccesarry 
        FLAG_COEFFS_BACK_WARP_SET = false;
        FLAG_COEFFS_FORWARD_WARP_SET = false;
        FLAG_MAPS_FORWARD_SET = false;
        FLAG_MAPS_BACK_WARP_SET = false;
        
}

void CThinPlateSpline::setCorrespondences(const std::vector<Point>& pS, const std::vector<Point>& pD)
{
        pSrc = pS;
        pDst = pD;

        // tell the class to recompute the coefficients if neccesarry 
        FLAG_COEFFS_BACK_WARP_SET = false;
        FLAG_COEFFS_FORWARD_WARP_SET = false;
        FLAG_MAPS_FORWARD_SET = false;
        FLAG_MAPS_BACK_WARP_SET = false;
}

double CThinPlateSpline::fktU(const Point& p1, const Point& p2) 
{
        double r = pow(((double)p1.x - (double)p2.x), 2) + pow(((double)p1.y - (double)p2.y), 2);

        if (r == 0)
                return 0.0;
        else 
        {
                //r = sqrt(r); // vector length
                //double r2 = pow(r, 2);

                return (r  * log(r));
        }
}

void CThinPlateSpline::computeSplineCoeffs(std::vector<Point>& iPIn, std::vector<Point>& iiPIn, float lambda,const TPS_INTERPOLATION tpsInter)
{

        std::vector<Point>* iP = NULL;
        std::vector<Point>*     iiP = NULL;

        if(tpsInter == FORWARD_WARP)
        {
                iP = &iPIn;
                iiP = &iiPIn;

                // keep information which coefficients are set
                FLAG_COEFFS_BACK_WARP_SET = true;
                FLAG_COEFFS_FORWARD_WARP_SET = false;
        }
        else if(tpsInter == BACK_WARP)
        {
                iP = &iiPIn;
                iiP = &iPIn;

                // keep information which coefficients are set
                FLAG_COEFFS_BACK_WARP_SET = false;
                FLAG_COEFFS_FORWARD_WARP_SET = true;
        }

        //get number of corresponding points
        int dim = 2;
        int n = iP->size();

        //Initialize mathematical datastructures
        Mat_<float> V(dim,n+dim+1,0.0);
        Mat_<float> P(n,dim+1,1.0);
        Mat_<float> K = (K.eye(n,n)*lambda);
        Mat_<float> L(n+dim+1,n+dim+1,0.0);

        // fill up K und P matrix
        std::vector<Point>::iterator itY;
        std::vector<Point>::iterator itX;
        
        int y = 0;
        for (itY = iP->begin(); itY != iP->end(); ++itY, y++) {
                int x = 0;
                for (itX = iP->begin(); itX != iP->end(); ++itX, x++) {
                        if (x != y) {
							    float temp = (float)fktU(*itY, *itX);
                                K(y, x) = (float)fktU(*itY, *itX);
                        }
                }
               
				P(y,1) = (float)itY->x;
				P(y,2) = (float) itY->y;

        }

        Mat Pt;
        transpose(P,Pt);

        // insert K into L
        Rect range = Rect(0, 0, n, n);
        Mat Lr(L,range);
        K.copyTo(Lr);


        // insert P into L
        range = Rect(n, 0, dim + 1, n);
        Lr = Mat(L,range);
        P.copyTo(Lr);

        // insert Pt into L
        range = Rect(0,n,n,dim+1);
        Lr = Mat(L,range);
        Pt.copyTo(Lr);

		/*for (int i = 0; i < L.rows; i++)
		{
			for (int j = 0; j < L.cols; j++)
			{
				std::cout<< L(i,j) <<"  ";
			}
			std::cout<< std::endl;
		}*/

        // fill V array
        std::vector<Point>::iterator it;
        int u = 0;

        for(it = iiP->begin(); it != iiP->end(); ++it)
        {
                V(0,u) = (float)it->x;
                V(1,u) = (float)it->y;
                u++;
        }

        // transpose V
        /*Mat Vt;
        transpose(V,Vt)*/;
		//showmat(Vt);
		//std::cout<<Vt<<std::endl;

        cMatrix = Mat_<float>(n+dim+1,dim,0.0);

        // invert L
        Mat invL;
        invert(L,invL,DECOMP_LU);

		//showmat(invL);
       // multiply(invL,Vt,cMatrix);
       cMatrix = invL * V.t();
	  // cvMatMul(invL,Vt,cMatrix);

		//showmat(cMatrix);

        //compensate for rounding errors
        for(int row = 0; row < cMatrix.rows; row++)
        {
                for(int col = 0; col < cMatrix.cols; col++)
                {
                        double v = cMatrix(row,col);
                        if(v > (-1.0e-006) && v < (1.0e-006) )
                        {
                                cMatrix(row,col) = 0.0;
                        }
                }
        }

		//showmat(cMatrix);
}


Point CThinPlateSpline::interpolate_forward_(const Point& p)
{
        Point2f interP;
        std::vector<Point>* pList = &pSrc;

        int k1 = cMatrix.rows - 3;
        int kx = cMatrix.rows - 2;
        int ky = cMatrix.rows - 1;

        double a1 = 0, ax = 0, ay = 0, cTmp = 0, uTmp = 0, tmp_i = 0, tmp_ii = 0;

        for (int i = 0; i < 2; i++) {
                a1 = cMatrix(k1,i);
                ax = cMatrix(kx,i);
                ay = cMatrix(ky,i);

                tmp_i = a1 + ax * p.y + ay * p.x;
                tmp_ii = 0;

                for (int j = 0; j < (int)pSrc.size(); j++) {
                        cTmp = cMatrix(j,i);
                        uTmp = fktU( (*pList)[j], p);

                        tmp_ii = tmp_ii + (cTmp * uTmp);
                }

                if (i == 0) {
                        interP.y = (float)(tmp_i + tmp_ii);
                }
                if (i == 1) {
                        interP.x = (float)(tmp_i + tmp_ii);
                }
        }

        return interP;
}
Point CThinPlateSpline::interpolate_back_(const Point& p)
{
        Point2f interP;
        std::vector<Point>* pList = &pDst;

        int k1 = cMatrix.rows - 3;
        int kx = cMatrix.rows - 2;
        int ky = cMatrix.rows - 1;

        double a1 = 0, ax = 0, ay = 0, cTmp = 0, uTmp = 0, tmp_i = 0, tmp_ii = 0;

        for (int i = 0; i < 2; i++) {
                a1 = cMatrix(k1,i);
                ax = cMatrix(kx,i);
                ay = cMatrix(ky,i);

                tmp_i = a1 + ax * p.y + ay * p.x;
                tmp_ii = 0;

                for (int j = 0; j < (int)pSrc.size(); j++) {
                        cTmp = cMatrix(j,i);
                        uTmp = fktU( (*pList)[j], p);

                        tmp_ii = tmp_ii + (cTmp * uTmp);
                }

                if (i == 0) {
                        interP.y = (float)(tmp_i + tmp_ii);
                }
                if (i == 1) {
                        interP.x = (float)(tmp_i + tmp_ii);
                }
        }

        return interP;
}

Point CThinPlateSpline::interpolate(const Point& p, const TPS_INTERPOLATION tpsInter)
{
        if(tpsInter == BACK_WARP)
        {
                return interpolate_back_(p);
        }
        else if(tpsInter == FORWARD_WARP)
        {
                return interpolate_forward_(p);
        }
        else
        {
                return interpolate_back_(p);
        }
        
}

void CThinPlateSpline::warpImage(const Mat& src, Mat& dst, float lambda, const int interpolation,const TPS_INTERPOLATION tpsInter)
{
        Size size = src.size();
        dst = Mat(size,src.type());

        // only compute the coefficients new if they weren't already computed
        // or there had been changes to the points
        if(tpsInter == BACK_WARP && !FLAG_COEFFS_BACK_WARP_SET)
        {
                computeSplineCoeffs(pSrc,pDst,lambda,tpsInter);
        }
        else if(tpsInter == FORWARD_WARP && !FLAG_COEFFS_FORWARD_WARP_SET)
        {
                computeSplineCoeffs(pSrc,pDst,lambda,tpsInter);
        }
        
        computeMaps(size,mapx,mapy);

        remap(src,dst,mapx,mapy,interpolation);
}

void CThinPlateSpline::getMaps(Mat& mx, Mat& my)
{
        mx = mapx;
        my = mapy;
}

void CThinPlateSpline::computeMaps(const Size& dstSize, Mat_<float>& mx, Mat_<float>& my,const TPS_INTERPOLATION tpsInter)
{
        mx = Mat_<float>(dstSize);
        my = Mat_<float>(dstSize);

        Point p(0, 0);
        Point_<float> intP(0, 0);
        
        for (int row = 0; row < dstSize.height; row++) {
                for (int col = 0; col < dstSize.width; col++) {
                        p = Point(col, row);
                        intP = interpolate(p,tpsInter);
                        mx(row, col) = intP.x;
                        my(row, col) = intP.y;
                }
        }

        if(tpsInter == BACK_WARP)
        {       
                FLAG_MAPS_BACK_WARP_SET = true;
                FLAG_MAPS_FORWARD_SET = false;
        }
        else if(tpsInter == FORWARD_WARP)
        {
                FLAG_MAPS_BACK_WARP_SET = false;
                FLAG_MAPS_FORWARD_SET = true;
        }
}

void CThinPlateSpline::pts(Mat src,Mat dst)
{
	//Size srcsize = src.size();
	int row = dst.rows;
	int cols = dst.cols;

	int srcrows = src.rows;
	int srccols = src.cols;
	
	//synImg = Mat(dst.rows,dst.cols,CV_8UC1);
	synImg = Mat(dst);

	Rect range = Rect(0,0,2,8);
	Mat w = Mat(cMatrix,range);
	range = Rect(0,8,2,3);
	Mat a = Mat(cMatrix,range);

	//showmat(w);
	//showmat(a);
	std::vector<cv::Point>::iterator itx;
	int max_x = 0;
	int min_x = 1000;
	int up = 0;
	int buttom = 1000;

	Mat_<float> dstipts(pDst.size(),2,1);

	int raws = 0;
	//vector<cv::Point>::iterator itx_s = pSrc.begin();

	//查找目标关键点的最大值和最小值(x,y坐标)
	for (itx = pDst.begin();itx != pDst.end();itx++)
	{
		max_x = (itx->x > max_x) ? itx->x : max_x;
		min_x = (itx->x < min_x) ? itx->x : min_x;


		up = (itx->y > up) ? itx->y : up;
		buttom = (itx->y < buttom) ? itx->y : buttom;

		dstipts(raws,0) = itx->x;
		dstipts(raws,1) = itx->y;
		raws++;

		
	}

	//根据最大和最小值，将目标区域几外扩展10%
	max_x += max_x / 20;
	min_x -= min_x / 20;
	up += up / 20;
	buttom -= buttom / 20;


    Mat_<float> u(row,2,1.0);
	for (int j = 1; j <= row; j++)
	{
		u(j-1,1) = j;
	}

	Mat_<float> mask(row,3,0.0);
	for (int i = buttom; i <up; i++)
	{
		mask(i,0) = 1;
		mask(i,1) = 1;
		mask(i,2) = 1;
	}
	//showmat(u);
	Mat_<float> tps;
	for (int i= min_x; i < max_x; i++)
	{
		tps = psi_tps(i,row,u,a,w,dstipts,mask);
		
		for (int ii = buttom; ii < up; ii++)
		{
			/*int x = tps(ii,0);
			int y = tps(ii,1);*/

			if ((tps(ii,0) > 0) &&
				(tps(ii,0) < srccols) && 
				(tps(ii,1) > 0) && 
				(tps(ii,1) < srcrows))
			{
				//synImg(ii,i,)
				synImg.at<Vec3b>(ii,i)[0] = src.at<Vec3b>(tps(ii,1),tps(ii,0))[0];
				synImg.at<Vec3b>(ii,i)[1] = src.at<Vec3b>(tps(ii,1),tps(ii,0))[1];
				synImg.at<Vec3b>(ii,i)[2] = src.at<Vec3b>(tps(ii,1),tps(ii,0))[2];

			}
		}
	}

	//cv::imshow("result",synImg);
	
}



Mat CThinPlateSpline::pts(Mat src,int width,int height)
{
	/*cv::imshow("src",src);
	cv::waitKey(0);*/
	//Size srcsize = src.size();
	int row = height;
	int cols = width;

	int srcrows = src.rows;
	int srccols = src.cols;
	
	//synImg = Mat(dst.rows,dst.cols,CV_8UC1);
	//synImg = Mat(dst);
	Mat wimg = Mat::zeros(height,width,CV_8UC1);
	int num = pSrc.size();
	Rect range = Rect(0,0,2, num);
	Mat w = Mat(cMatrix,range);
	range = Rect(0, num,2,3);
	Mat a = Mat(cMatrix,range);

	//showmat(w);
	//showmat(a);
	std::vector<cv::Point>::iterator itx;
	int max_x = 0;
	int min_x = 1000;
	int up = 0;
	int buttom = 1000;

	Mat_<float> dstipts(pDst.size(),2,1);

	int raws = 0;
	//vector<cv::Point>::iterator itx_s = pSrc.begin();

	//查找目标关键点的最大值和最小值(x,y坐标)
	for (itx = pDst.begin();itx != pDst.end();itx++)
	{
		max_x = (itx->x > max_x) ? itx->x : max_x;
		min_x = (itx->x < min_x) ? itx->x : min_x;


		up = (itx->y > up) ? itx->y : up;
		buttom = (itx->y < buttom) ? itx->y : buttom;

		dstipts(raws,0) = itx->x;
		dstipts(raws,1) = itx->y;
		raws++;

		
	}

	//根据最大和最小值，将目标区域几外扩展10%
	float edge = 0.2;
	max_x += max_x * edge;
	min_x -= min_x * edge;
	up += up * edge;
	buttom -= buttom * edge;

	if(min_x < 0)
	{
		min_x = 0;
	}

	if(buttom < 0)
	{
		buttom = 0;
	}

	if(max_x >= row)
	{
		max_x = row - 1;
	}

	if(up >= cols)
	{
		up = cols - 1;
	}

    Mat_<float> u(row,2,1.0);
	for (int j = 1; j <= row; j++)
	{
		u(j-1,1) = j;
	}

	Mat_<float> mask(row,3,0.0);
	for (int i = buttom; i <up; i++)
	{
		mask(i,0) = 1;
		mask(i,1) = 1;
		mask(i,2) = 1;
	}
	//showmat(u);
	Mat_<float> tps;
	for (int i= min_x; i < max_x; i++)
	{
		tps = psi_tps(i,row,u,a,w,dstipts,mask);
		
		for (int ii = buttom; ii < up; ii++)
		{
			/*int x = tps(ii,0);
			int y = tps(ii,1);*/

			if ((tps(ii,0) > 0) &&
				(tps(ii,0) < srccols) && 
				(tps(ii,1) > 0) && 
				(tps(ii,1) < srcrows))
			{
				
				int x = tps(ii,1);
				int y = tps(ii,0);
				uchar va = src.ptr<uchar>(x)[y];
				//wimg.ptr<uchar>(ii)[i] = va;
				if(va>128)
				{
					va = 255;
					wimg.ptr<uchar>(ii)[i] = va;
				}
				//std::cout << va << std::endl;
				//wimg.ptr<uchar>(ii)[i] = va;

			}
		}
	}

	//cv::imshow("result",synImg);
	return wimg;
	
}

Mat_<float> CThinPlateSpline::psi_tps(int col,int row,Mat_<float>& u,Mat& a, Mat& w,Mat& ipts,Mat_<float> Mask)
{
	Mat_<float> temp(row,1,col);  
	Rect range = Rect(0,0,1,row);
	Mat cu = Mat(u,range);
	temp.copyTo(cu);
	//showmat(u);
	
	Mat_<float> ones(row,3,1.0);
	range = Rect(1,0,2,row);
	/*Mat_<float> temp = Mat_(row,1,col);
	
	temp.copyTo(cyones);*/

	Mat cyones = Mat(ones,range);
	
	cyones = Mat(ones,range);
	u.copyTo(cyones);
   // showmat(ones);
	ones = ones.mul(Mask);
	
	range = Rect(1,0,2,Mask.rows);
	Mat cMask = Mat(Mask,range);
	u = u.mul(cMask);
	
	return (ones * a + matdistance(u.t(),ipts.t()) * w);

}


void CThinPlateSpline::initImg(Size msize)
{
	synImg = Mat(msize,CV_8U);
}




Mat_<float> CThinPlateSpline::matdistance(Mat_<float> a, Mat_<float> b)
{
	if (a.rows != b.rows)
	{
		printf("a and b should be of same dimesionality");
		//return;
	}
	

	

	Mat_<float> tempa = a.mul(a);
	Mat_<float> aa;
	reduce(tempa,aa,0,CV_REDUCE_SUM);  //sum

	//showmat(aa);

	Mat_<float> tempb = b.mul(b);
	Mat_<float> bb;
	reduce(tempb,bb,0,CV_REDUCE_SUM);  //sum

	//showmat(bb);
	/*Mat_<float> aT;
	transpose(a,aT);*/
	
	Mat_<float> ab = a.t() * b;
	//showmat(ab);
	/*Mat_<float> aaT;
	transpose(aa,aaT);*/

    Mat_<float> repa;
	repeat(aa.t(),1,bb.cols,repa);
	//showmat(repa);
	Mat_<float> repb;
	repeat(bb,aa.cols,1,repb);
//	showmat(repb);
	Mat_<float> d;
	//sqrt(abs(repa + repb - 2*ab),d);
	Mat_<float> r = abs(repa + repb - 2*ab);
  //  showmat(r);
	Mat_<float> logr;
	log(r+1,logr);

	d = r.mul(logr);

	return d;

}

void CThinPlateSpline::showmat(Mat_<float> mat)
{
	//std::fstream ff;
	//ff.open("mat.txt",std::ios::in | std::ios::out);
	
	for (int i=0; i < mat.rows; i++)
	{
		for (int j=0; j <mat.cols; j++)
		{
			std::cout<<mat(i,j)<<" ";
		//	ff << mat(i,j) << " ";
		}
		std::cout<<std::endl;
		//ff << "\n";
	}
	//ff.close();
}