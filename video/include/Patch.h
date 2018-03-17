

#ifndef __Patch_h_
#define __Patch_h_


#include <opencv2/core/core.hpp>


#include "SVR_patch.h"
#include "CCNF_patch_expert.h"
#include "FPD.h"

namespace KeyPointsDetector
{

class Patch
{

public:

	// The collection of SVR patch experts (for intensity/grayscale images), the experts are laid out scale->view->landmark
	vector<vector<vector<Multi_SVR_patch_expert> > >	svr_expert_intensity;
	 
	// The collection of SVR patch experts (for depth/range images), the experts are laid out scale->view->landmark
	vector<vector<vector<Multi_SVR_patch_expert> > >	svr_expert_depth;

	// The collection of LNF (CCNF) patch experts (for intensity images), the experts are laid out scale->view->landmark
	vector<vector<vector<CCNF_patch_expert> > >			ccnf_expert_intensity;

	// The node connectivity for CCNF experts, at different window sizes and corresponding to separate edge features
	vector<vector<cv::Mat_<float> > >					sigma_components;

	// The available scales for intensity patch experts
	vector<double>							patch_scaling;

	// The available views for the patch experts at every scale (in radians)
	vector<vector<cv::Vec3d> >               centers;

	// Landmark visibilities for each scale and view
    vector<vector<cv::Mat_<int> > >          visibilities;

	// A default constructor
	Patch(){;}

	// A copy constructor
	Patch(const Patch& other);

	
	void Response(vector<cv::Mat_<float> >& patch_expert_responses, cv::Matx22f& sim_ref_to_img, cv::Matx22d& sim_img_to_ref, const cv::Mat_<uchar>& grayscale_image, const cv::Mat_<float>& depth_image,
							 const FPD& FPD, const cv::Vec6d& params_global, const cv::Mat_<double>& params_local, int window_size, int scale);

	// Getting the best view associated with the current orientation
	int GetViewIdx(const cv::Vec6d& params_global, int scale) const;

	// The number of views at a particular scale
	inline int nViews(int scale = 0) const { return centers[scale].size(); };

	// Reading in all of the patch experts
	void Read(vector<string> intensity_svr_expert_locations, vector<string> depth_svr_expert_locations, vector<string> intensity_ccnf_expert_locations);


   

private:
	void Read_SVR_Patch(string expert_location, std::vector<cv::Vec3d>& centers, std::vector<cv::Mat_<int> >& visibility, std::vector<std::vector<Multi_SVR_patch_expert> >& patches, double& scale);
	void Read_CCNF_Patch(string patchesFileLocation, std::vector<cv::Vec3d>& centers, std::vector<cv::Mat_<int> >& visibility, std::vector<std::vector<CCNF_patch_expert> >& patches, double& patchScaling);
	

};
 
}
#endif
