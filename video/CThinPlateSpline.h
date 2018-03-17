

#ifndef CTHINPLATESPLINE_H_
#define CTHINPLATESPLINE_H_
using namespace cv;


enum TPS_INTERPOLATION
{
        FORWARD_WARP,                                   /**< forward transformation (optional)  */
        BACK_WARP                                               /**< back warp transform (standard)             */
};


class CThinPlateSpline {
public:
        
        CThinPlateSpline();


       
        CThinPlateSpline(const std::vector<Point>& pS, const std::vector<Point>& pD);


        

        ~CThinPlateSpline();

      
        Point interpolate(const Point& p, const TPS_INTERPOLATION tpsInter = BACK_WARP);

       
        void addCorrespondence(const Point& pS, const Point& pD);

       
        void warpImage(const Mat& src, 
                Mat& dst, 
                float lambda = 0.001, 
                const int interpolation = INTER_CUBIC, 
                const TPS_INTERPOLATION tpsInter = BACK_WARP);


       
        void setCorrespondences(const std::vector<Point>& pS, const std::vector<Point>& pD);
        
        
        void getMaps(Mat& mapx, Mat& mapy);

       
        void computeMaps(const Size& dstSize, 
                Mat_<float>& mapx, 
                Mat_<float>& mapy,
                const TPS_INTERPOLATION tpsInter = BACK_WARP);

		Mat_<float> psi_tps(int col,int row,Mat_<float>& u,Mat& a, Mat& w,Mat& ipts,Mat_<float> Mask);
		void initImg(Size msize);
		void pts(Mat src,Mat dst);
		Mat pts(Mat src,int width,int height);
		Mat_<float> matdistance(Mat_<float> a, Mat_<float> b);
		void computeSplineCoeffs(std::vector<Point>& iP,
			std::vector<Point>& iiP, 
			float lambda, 
			const TPS_INTERPOLATION tpsInter = BACK_WARP);

		void showmat(Mat_<float> mat);
		Mat synImg;

private:

        
        double fktU(const Point& p1, const Point& p2);

        

        Point interpolate_back_(const Point& p);
        Point interpolate_forward_(const Point& p);

		

        Mat_<float> cMatrix;
        Mat_<float> mapx;
        Mat_<float> mapy;
        std::vector<Point> pSrc;
        std::vector<Point> pDst;

		//Mat synImg;

        // FLAGS
        bool FLAG_COEFFS_BACK_WARP_SET;
        bool FLAG_COEFFS_FORWARD_WARP_SET;
        bool FLAG_MAPS_FORWARD_SET;
        bool FLAG_MAPS_BACK_WARP_SET;

};

#endif /* CTHINPLATESPLINE_H_ */