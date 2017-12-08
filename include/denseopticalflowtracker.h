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
    cv::Ptr<cv::cuda::BroxOpticalFlow> brox ;
    cv::Ptr<cv::cuda::DensePyrLKOpticalFlow> lk ;
    cv::Ptr<cv::cuda::FarnebackOpticalFlow> farn ;
    cv::Ptr<cv::cuda::OpticalFlowDual_TVL1> tvl1;

    cv::cuda::GpuMat d_flow_forward;
    cv::cuda::GpuMat d_flow_backward;
    cv::Mat prvFrame,currentFrame;
    cv::Mat_<float> flowx_forward, flowy_forward;
public:
    denseOpticalFlowTracker(cv::Size sz, std::string);

    void trackPoints(const cv::Mat &prvFrame,
                     const cv::Mat &currentFrame,
                     std::vector<cv::Point2f> &prvFramePoints,
                     std::vector<cv::Point2f> &currentFramePoints,
                     bool reinit);


    void trackPoints(const cv::Mat &prvFrame,
                     const cv::Mat &currentFrame,
                     corr &prev_correspondance,
                     corr &current_correspondance,
                     bool reinit);

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
    cv::Rect2d getReliablePoints(const cv::Mat & frame0,
                             const cv::Mat & frame1,
                             const cv::cuda::GpuMat & forwardFlow,
                             const cv::cuda::GpuMat & backwardFlow,
                             std::vector<cv::Point2f> &startPoints,
                             std::vector<cv::Point2f> &trackedPoints,
                             cv::Mat_<float> &flowx_forward,
                             cv::Mat_<float> &flowy_forward,
                             cv::Rect2d inpRect);

    cv::Rect2d predictBoundingBox(const std::vector<cv::Point2f> &startPoints,
                                  const std::vector<cv::Point2f> &trackedPoints,
                                  cv::Rect2d bb);
};

#endif // DENSEOPTICALFLOWTRACKER_H
