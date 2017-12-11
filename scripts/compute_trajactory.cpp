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


DEFINE_string(dirname, "/home/terminalx/data/Freeman1/imgs", "Directory to dump in");
DEFINE_string(video, "/home/terminalx/data/Freeman1/freeman1.mp4", "Name of the video");
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

cv::Vec3f detectROI(const Mat &img)
{
    Vec3f resResCircle;

    Mat gray, dst;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;


    cv::cvtColor( img, gray, CV_BGR2GRAY );
    cv::GaussianBlur( gray, gray, Size(9, 9), 2, 2 );
    cv::threshold( gray, dst, 10, 255, 0 );

    imshow("dst", dst);
    findContours( dst, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );


    Point2f center;
    float radius;
    vector<Point> contours_poly;
    if(contours.size())
    {
        vector<Point> maxContour = contours[0];
        for( int i = 1; i < contours.size(); i++ )
        {
            if(contours[i].size() > maxContour.size())
            {
                maxContour = contours[i];
            }
        }

        approxPolyDP( Mat(maxContour), contours_poly, 3, true );
        minEnclosingCircle( (Mat)contours_poly, center, radius);

        resResCircle[0] = (float)center.x;
        resResCircle[1] = (float)center.y;
        resResCircle[2] = radius;
    }

    return resResCircle;
}


inline vector<Rect2d> readGT(const string &filename)
{
    vector<Rect2d> res;
    {
        ifstream input(filename.c_str());
        if (!input.is_open())
            CV_Error(Error::StsError, "Failed to open file");
        while (input)
        {
            Rect2d one;
            input >> one.x;
            input.ignore(numeric_limits<std::streamsize>::max(), ',');
            input >> one.y;
            input.ignore(numeric_limits<std::streamsize>::max(), ',');
            input >> one.width;
            input.ignore(numeric_limits<std::streamsize>::max(), ',');
            input >> one.height;
            input.ignore(numeric_limits<std::streamsize>::max(), '\n');
            if (input.good())
            {
                //one.width += 50;
                //one.height += 120;
                res.push_back(one);
            }

        }
    }
    return res;
}

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
    Mat currentFrame, prvFrame, drawing;
    int cx, cy, focal;

    vector<Rect2d> resultGT = readGT("/home/terminalx/data/Freeman1/groundtruth_rect.txt");
    int gtCount = 0;
    Rect2d trackedRect, prvTrackedRect;
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
    //for(int i = 0; i < 1000; i++)
    //{
    //    vidCapture >> prvFrame;
    //}
    /*----------------------------------------------------------------------------------------------------*/

    /*----------------------------------------------------------------------------------------------------*/
    /*!
     * \brief create windows and initalize the dense video tracker
     */
    //namedWindow("input-video",1);
    namedWindow("drawing",1);
    vidCapture >> prvFrame;
    denseOpticalFlowTracker *denseTracker = new denseOpticalFlowTracker(cv::Size(prvFrame.cols, prvFrame.rows) , "brox");
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
//            prvFrameCorrespondance.p1.clear();
//            prvFrameCorrespondance.p2.clear();
//            if(m_readCount == 1)
//            {
//                 denseTracker->trackCorrespondance(prvFrame, currentFrame, prvFrameCorrespondance, currentFrameCorrespondance, true, totalFeatures, resultGT[gtCount], prvTrackedRect);
//            }
//            else
//            {
//                denseTracker->trackCorrespondance(prvFrame, currentFrame, prvFrameCorrespondance, currentFrameCorrespondance, true, totalFeatures, prvTrackedRect, trackedRect);
//                prvTrackedRect = trackedRect;
//            }

            if((m_readCount-1) % 20 == 0 || currentFrameCorrespondance.p2.size()  < FLAGS_min_corners)
            {
                denseTracker->trackCorrespondance(prvFrame, currentFrame, prvFrameCorrespondance, currentFrameCorrespondance, true, totalFeatures, cv::Rect2d(), prvTrackedRect);
            }
            else
            {
                denseTracker->trackCorrespondance(prvFrame, currentFrame, prvFrameCorrespondance, currentFrameCorrespondance, false, totalFeatures, cv::Rect2d(), prvTrackedRect);
            }
            cout << "current frame correspondances - " << currentFrameCorrespondance.p1.size() << std::endl;
            allFramesCorrespondance.push_back(currentFrameCorrespondance);
            prvFrameCorrespondance.p1 = prvFrameCorrespondance.p2;
            prvFrameCorrespondance = currentFrameCorrespondance;

            currentFrame.copyTo(prvFrame);

            for (int i=0; i<prvFrameCorrespondance.p1.size(); i++)
            {
              cv::circle(drawing, prvFrameCorrespondance.p1[i], 4, cv::Scalar(0, 255, 0), -1);
            }

            //cv::Vec3f cir = detectROI(currentFrame);
            //circle( drawing, Point(cir[0], cir[1]), cir[2], Scalar(0,0,255), 3, 8, 0 );

            //rectangle(drawing, resultGT[++gtCount], Scalar(0,0,255), 3);
            //rectangle(drawing, prvTrackedRect, Scalar(0,255,0), 2);
            cv::imshow("drawing", drawing);
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
    delete denseTracker;
    return 0;
}
