#include <iostream>
#include <fstream>

#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/video.hpp"
#include "opencv2/optflow.hpp"
#include "opencv2/core/ocl.hpp"
#include <iomanip>
#include <fstream>
#include <limits>
#include "correspondance.h"
#include "denseopticalflowtracker.h"
#include "FishCam.h"
#include <gflags/gflags.h>


DEFINE_string(dirname, "/home/terminalx/britty/data/OR_11_Nov_17/SLAM_P/imgs", "Directory to dump in");
DEFINE_string(video, "/home/terminalx/britty/data/OR_11_Nov_17/SLAM_P/L_p.avi", "Name of the video");
DEFINE_string(calib, "/home/terminalx/britty/data/OR_11_Nov_17/SLAM_P/calib_results.txt", "Calibration file");
DEFINE_int32(keyframe, 10, "Max number of frames in a keyframe");
DEFINE_int32(chunks, 150, "Max number of keyframes in a chunk");
DEFINE_int32(overlap, 30, "Number of frames to be considered in the overalp");
DEFINE_int32(min_corners, 100, "Minimum number of points in image below which more will be added");
DEFINE_bool(corres, false, "Dump image correspondances");
DEFINE_bool(undistort, false, "Undistort the images");
DEFINE_bool(use_sift, false, "Use sift for corresponances");

using namespace std;
using namespace cv;
using namespace cv::cuda;

int main(int argc, const char* argv[])
{
    /*----------------------------------------------------------------------------------------------------*/
    /*!
     * \brief declare all the variables
     */
    bool paused = false;
    int m_frameH, m_frameW, m_fpsVideo, m_numFrames;
    int m_readCount = 0;
    int totalFeatures = 0;
    Rect2d trackedRect;
    int gtCount = 0;
    Mat currentFrame, prvFrame, drawing;
    int cx, cy, focal;
    /*----------------------------------------------------------------------------------------------------*/


    /*----------------------------------------------------------------------------------------------------*/
    /*!
     * \brief declare all the variables
     */
    FishOcam cam_model;
    if (FLAGS_undistort)
    {
        cam_model.init(FLAGS_calib);
        focal = cam_model.focal;
        std::cout << "Focal " << focal << "\n";
    }
    /*----------------------------------------------------------------------------------------------------*/


    /*----------------------------------------------------------------------------------------------------*/
    /*!
     * \brief vidCapture stuff
     */
    VideoCapture vidCapture(FLAGS_video);
    //VideoCapture vidCapture("/home/terminalx/data/Freeman1/freeman1.mp4");
    if(!vidCapture.isOpened())
        return -1;
    m_frameH = vidCapture.get(CV_CAP_PROP_FRAME_HEIGHT);
    m_frameW = vidCapture.get(CV_CAP_PROP_FRAME_WIDTH);
    m_fpsVideo = vidCapture.get(CV_CAP_PROP_FPS);
    m_numFrames = vidCapture.get(CV_CAP_PROP_FRAME_COUNT);
    for(int i = 0; i < 1000; i++)
    {
        vidCapture >> prvFrame;
    }
    /*----------------------------------------------------------------------------------------------------*/

    /*----------------------------------------------------------------------------------------------------*/
    /*!
     * \brief create windows and initalize the dense video tracker
     */
    namedWindow("input-video",1);
    namedWindow("drawing",1);
    vidCapture >> prvFrame;
    denseOpticalFlowTracker denseTracker(prvFrame.size(), "brox");
    cx = prvFrame.cols/2;
    cy = prvFrame.rows/2;
    if (!FLAGS_undistort)
    {
      focal *= 2*cx;
    }
    corr prvFrameCorrespondance(m_readCount-1, m_readCount);
    corr currentFrameCorrespondance;
    std::vector<corr> allFramesCorrespondance;
    /*----------------------------------------------------------------------------------------------------*/

    for(;;)
    {
        if(!paused)
        {
            m_readCount++;
            vidCapture >> currentFrame;
            currentFrame.copyTo(drawing);
            if(currentFrame.empty())
                break;
            if (FLAGS_undistort)
            {
              cv::Mat undistorted;
              cam_model.WarpImage(currentFrame, undistorted);
              // undistort(rawFrame);
              undistorted.copyTo(currentFrame);
            }
            if((m_readCount-1) % 5 == 0 || currentFrameCorrespondance.p2.size()  < FLAGS_min_corners)
            {
                currentFrameCorrespondance = denseTracker.trackCorrespondance(prvFrame, currentFrame, prvFrameCorrespondance, totalFeatures, true);
                totalFeatures += currentFrameCorrespondance.unique_id.size();
            }
            else
            {
                currentFrameCorrespondance = denseTracker.trackCorrespondance(prvFrame, currentFrame, prvFrameCorrespondance, totalFeatures, false);
            }
            allFramesCorrespondance.push_back(currentFrameCorrespondance);
            prvFrameCorrespondance = currentFrameCorrespondance;


            currentFrame.copyTo(prvFrame);
            imshow("drawing", drawing);
            char ch = waitKey(2);
            if(ch == 'q')
            {
                break;
            }
            else if(ch == 'p')
            {
                paused = !paused;
            }
        }
    }

    return 0;
}
