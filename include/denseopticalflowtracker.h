#ifndef DENSEOPTICALFLOWTRACKER_H
#define DENSEOPTICALFLOWTRACKER_H
#include <string>

#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/video.hpp"
#include "opencv2/optflow.hpp"
#include "opencv2/core/ocl.hpp"
#include "correspondance.h"

class denseOpticalFlowTracker
{
private:
    std::string method;


    //cv::cuda::GpuMat forwardFlow;
    //cv::cuda::GpuMat backwardFlow;
    cv::Mat prvFrame,currentFrame, prvFrameGray,currentFrameGray ;
    cv::Mat_<float> flowx_forward;
    cv::Mat_<float> flowy_forward;
    cv::Mat_<float> flowx_backward;
    cv::Mat_<float> flowy_backward;
    cv::cuda::GpuMat framegpu0f, framegpu1f;

    cv::Ptr<cv::cuda::BroxOpticalFlow> brox ;
    cv::Ptr<cv::cuda::DensePyrLKOpticalFlow> lk ;
    cv::Ptr<cv::cuda::FarnebackOpticalFlow> farn ;
    cv::Ptr<cv::cuda::OpticalFlowDual_TVL1> tvl1;

public:
    denseOpticalFlowTracker(cv::Size sz, std::string);

    void trackCorrespondance(const cv::Mat & frame0,
                                   const cv::Mat & frame1,
                                   const corr &previous_correspondance,
                                   corr &currentFrameCorrespondance,
                                   bool reinit,
                                   int &totalFeatures,
                                   cv::Rect2d inpRect,
                                   cv::Rect2d &resultBB);

    struct comparator_cvPoint2f
    {
        template<typename T, typename U>
        bool operator()(T const& lhs, U const& rhs) const {
            return lhs.x < rhs.x;
        }
    };

private:
    inline bool isFlowCorrect(cv::Point2f u);
    void elementSwap(std::vector<float> &vec, int N1, int N2);
    float computeMedian(const std::vector<float> &arr1);
    float computeMedian(const cv::Mat_<float> & input);


    cv::Rect2d predictBoundingBox(const std::vector<cv::Point2f> &startPoints,
                                  const std::vector<cv::Point2f> &trackedPoints,
                                  cv::Rect2d bb);

    void showFlow(const char* name, const cv::cuda::GpuMat& d_flow);
    void drawOpticalFlow(const cv::Mat_<float>& flowx, const cv::Mat_<float>& flowy, cv::Mat& dst, float maxmotion);
    cv::Vec3b computeColor(float fx, float fy);
};

#endif // DENSEOPTICALFLOWTRACKER_H
