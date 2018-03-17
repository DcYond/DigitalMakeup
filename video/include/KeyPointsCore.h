

#ifndef __KEY_POINTS_DETECTOR_CORE_h_
#define __KEY_POINTS_DETECTOR_CORE_h_

// OpenCV dependencies
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect.hpp>

#include "FPD.h"
#include "Patch.h"
#include "KeyPointsValidator.h"
#include "KeyPointsParameters.h"

using namespace std;

namespace KeyPointsDetector
{


class FACE
{

public:

	
    FPD			fpd;
	Patch		patch;

	// The local and global parameters describing the current model instance (current landmark detections)

	// Local parameters describing the non-rigid shape
	cv::Mat_<double>    params_local;

	// Global parameters describing the rigid shape [scale, euler_x, euler_y, euler_z, tx, ty]
	cv::Vec6d           params_global;

	// A collection of hierarchical FACE models that can be used for refinement
	vector<FACE>					hierarchical_models;
	vector<string>					hierarchical_model_names;
	vector<vector<pair<int,int>>>	hierarchical_mapping;
	vector<FaceModelParameters>		hierarchical_params;


	// Haar cascade classifier for face detection
	cv::CascadeClassifier   face_detector_HAAR;
	string                  face_detector_location;

	

	Validator	keypoints_validator; 

	// Indicating if landmark detection succeeded (based on SVR validator)
	bool				detection_success; 

	// Indicating if the tracking has been initialised (for video based tracking)
	bool				tracking_initialised;

	// The actual output of the regressor (-1 is perfect detection 1 is worst detection)
	double				detection_certainty; 

	// Indicator if eye model is there for eye detection
	bool				eye_model = false;

	// the triangulation per each view (for drawing purposes only)
	vector<cv::Mat_<int> >	triangulations;
	

	// Lastly detect 2D model shape [x1,x2,...xn,y1,...yn]
	cv::Mat_<double>		detected_keypoints;
	
	// The landmark detection likelihoods (combined and per patch expert)
	double				model_likelihood;
	cv::Mat_<double>		keypoints_likelihoods;
	
	// Keeping track of how many frames the tracker has failed in so far when tracking in videos
	// This is useful for knowing when to initialise and reinitialise tracking
	int failures_in_a_row;

	// A template of a face that last succeeded with tracking (useful for large motions in video)
	cv::Mat_<uchar> face_template;

	// Useful when resetting or initialising the model closer to a specific location (when multiple faces are present)
	cv::Point_<double> preference_det;

	string proot;
	// A default constructor
	FACE();

	// Constructor from a model file
	FACE(string fname);
	
	// Copy constructor (makes a deep copy of the detector)
	FACE(const FACE& other);

	// Assignment operator for lvalues (makes a deep copy of the detector)
	FACE & operator= (const FACE& other);

	// Empty Destructor	as the memory of every object will be managed by the corresponding libraries (no pointers)
	~FACE(){}

	// Move constructor
	FACE(const FACE&& other);

	// Assignment operator for rvalues
	FACE & operator= (const FACE&& other);

	// Does the actual work - landmark detection
	bool DetectKeyPoints(const cv::Mat_<uchar> &image, const cv::Mat_<float> &depth, FaceModelParameters& params);
	
	// Gets the shape of the current detected landmarks in camera space (given camera calibration)
	// Can only be called after a call to DetectKeyPointsInVideo or DetectKeyPointsInImage
	cv::Mat_<double> GetShape(double fx, double fy, double cx, double cy) const;

	// A utility bounding box function
	cv::Rect_<double> GetBoundingBox() const;

	// Reset the model (useful if we want to completelly reinitialise, or we want to track another video)
	void Reset();

	// Reset the model, choosing the face nearest (x,y) where x and y are between 0 and 1.
	void Reset(double x, double y);

	// Reading the model in
	void LoadModel(string name);

	
private:

	map<int, cv::Mat_<float> >		kde_resp_precalc;

	// The model fitting: patch response computation and optimisation steps
    bool Fit(const cv::Mat_<uchar>& intensity_image, const cv::Mat_<float>& depth_image, const std::vector<int>& window_sizes, const FaceModelParameters& parameters);

	// Mean shift computation that uses precalculated kernel density estimators (the one actually used)
	void NonVectorisedMeanShift_precalc_kde(cv::Mat_<float>& out_mean_shifts, const vector<cv::Mat_<float> >& patch_expert_responses, const cv::Mat_<float> &dxs, const cv::Mat_<float> &dys, int resp_size, float a, int scale, int view_id, map<int, cv::Mat_<float> >& mean_shifts);

	// The actual model optimisation (update step), returns the model likelihood
    double NU_RLMS(cv::Vec6d& final_global, cv::Mat_<double>& final_local, const vector<cv::Mat_<float> >& patch_expert_responses, const cv::Vec6d& initial_global, const cv::Mat_<double>& initial_local,
		          const cv::Mat_<double>& base_shape, const cv::Matx22d& sim_img_to_ref, const cv::Matx22f& sim_ref_to_img, int resp_size, int view_idx, bool rigid, int scale, cv::Mat_<double>& landmark_lhoods, const FaceModelParameters& parameters);

	// Removing background image from the depth
	bool RemoveBackground(cv::Mat_<float>& out_depth_image, const cv::Mat_<float>& depth_image);

	// Generating the weight matrix for the Weighted least squares
	void GetWeightMatrix(cv::Mat_<float>& WeightMatrix, int scale, int view_id, const FaceModelParameters& parameters);

	
  };
 
}
#endif
