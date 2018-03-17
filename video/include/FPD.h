

#ifndef __FPD_h_
#define __FPD_h_

// OpenCV includes
#include <opencv2/core/core.hpp>

#include "KeyPointsParameters.h"

namespace KeyPointsDetector
{

class FPD
{
	public:    
    
		// The 3D mean shape vector of the FPD [x1,..,xn,y1,...yn,z1,...,zn]
		cv::Mat_<double> mean_shape;	
  
		// Principal components or variation bases of the model, 
		cv::Mat_<double> princ_comp;	

		// Eigenvalues (variances) corresponding to the bases
		cv::Mat_<double> eigen_values;	

		FPD(){;}
		
		// A copy constructor
		FPD(const FPD& other);
			
		void Read(string location);

		// Number of vertices
		inline int NumberOfPoints() const 
		{return mean_shape.rows/3;}
		
		// Listing the number of modes of variation
		inline int NumberOfModes() const 
		{return princ_comp.cols;}

		void Clamp(cv::Mat_<float>& params_local, cv::Vec6d& params_global, const FaceModelParameters& params);

		// Compute shape in object space (3D)
		void CalcShape3D(cv::Mat_<double>& out_shape, const cv::Mat_<double>& params_local) const;

		// Compute shape in image space (2D)
		void CalcShape2D(cv::Mat_<double>& out_shape, const cv::Mat_<double>& params_local, const cv::Vec6d& params_global) const;
    
		// provided the bounding box of a face and the local parameters (with optional rotation), generates the global parameters that can generate the face with the provided bounding box
		void CalcParams(cv::Vec6d& out_params_global, const cv::Rect_<double>& bounding_box, const cv::Mat_<double>& params_local, const cv::Vec3d rotation = cv::Vec3d(0.0));

		// Provided the landmark location compute global and local parameters best fitting it (can provide optional rotation for potentially better results)
		void CalcParams(cv::Vec6d& out_params_global, const cv::Mat_<double>& out_params_local, const cv::Mat_<double>& landmark_locations, const cv::Vec3d rotation = cv::Vec3d(0.0));

		// provided the model parameters, compute the bounding box of a face
		void CalcBoundingBox(cv::Rect& out_bounding_box, const cv::Vec6d& params_global, const cv::Mat_<double>& params_local);

		// Helpers for computing Jacobians, and Jacobians with the weight matrix
		void ComputeRigidJacobian(const cv::Mat_<float>& params_local, const cv::Vec6d& params_global, cv::Mat_<float> &Jacob, const cv::Mat_<float> W, cv::Mat_<float> &Jacob_t_w);
		void ComputeJacobian(const cv::Mat_<float>& params_local, const cv::Vec6d& params_global, cv::Mat_<float> &Jacobian, const cv::Mat_<float> W, cv::Mat_<float> &Jacob_t_w);

		// Given the current parameters, and the computed delta_p compute the updated parameters
		void UpdateModelParameters(const cv::Mat_<float>& delta_p, cv::Mat_<float>& params_local, cv::Vec6d& params_global);

  };
  //===========================================================================
}
#endif
