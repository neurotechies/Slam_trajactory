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

using namespace std;
using namespace cv;
using namespace cv::cuda;

inline bool isFlowCorrect(Point2f u)
{
    return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9;
}

static Vec3b computeColor(float fx, float fy)
{
    static bool first = true;

    // relative lengths of color transitions:
    // these are chosen based on perceptual similarity
    // (e.g. one can distinguish more shades between red and yellow
    //  than between yellow and green)
    const int RY = 15;
    const int YG = 6;
    const int GC = 4;
    const int CB = 11;
    const int BM = 13;
    const int MR = 6;
    const int NCOLS = RY + YG + GC + CB + BM + MR;
    static Vec3i colorWheel[NCOLS];

    if (first)
    {
        int k = 0;

        for (int i = 0; i < RY; ++i, ++k)
            colorWheel[k] = Vec3i(255, 255 * i / RY, 0);

        for (int i = 0; i < YG; ++i, ++k)
            colorWheel[k] = Vec3i(255 - 255 * i / YG, 255, 0);

        for (int i = 0; i < GC; ++i, ++k)
            colorWheel[k] = Vec3i(0, 255, 255 * i / GC);

        for (int i = 0; i < CB; ++i, ++k)
            colorWheel[k] = Vec3i(0, 255 - 255 * i / CB, 255);

        for (int i = 0; i < BM; ++i, ++k)
            colorWheel[k] = Vec3i(255 * i / BM, 0, 255);

        for (int i = 0; i < MR; ++i, ++k)
            colorWheel[k] = Vec3i(255, 0, 255 - 255 * i / MR);

        first = false;
    }

    const float rad = sqrt(fx * fx + fy * fy);
    const float a = atan2(-fy, -fx) / (float) CV_PI;

    const float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);
    const int k0 = static_cast<int>(fk);
    const int k1 = (k0 + 1) % NCOLS;
    const float f = fk - k0;

    Vec3b pix;

    for (int b = 0; b < 3; b++)
    {
        const float col0 = colorWheel[k0][b] / 255.0f;
        const float col1 = colorWheel[k1][b] / 255.0f;

        float col = (1 - f) * col0 + f * col1;

        if (rad <= 1)
            col = 1 - rad * (1 - col); // increase saturation with radius
        else
            col *= .75; // out of range

        pix[2 - b] = static_cast<uchar>(255.0 * col);
    }

    return pix;
}

static void drawOpticalFlow(const Mat_<float>& flowx, const Mat_<float>& flowy, Mat& dst, float maxmotion = -1)
{
    dst.create(flowx.size(), CV_8UC3);
    dst.setTo(Scalar::all(0));

    // determine motion range:
    float maxrad = maxmotion;

    if (maxmotion <= 0)
    {
        maxrad = 1;
        for (int y = 0; y < flowx.rows; ++y)
        {
            for (int x = 0; x < flowx.cols; ++x)
            {
                Point2f u(flowx(y, x), flowy(y, x));

                if (!isFlowCorrect(u))
                    continue;

                maxrad = max(maxrad, sqrt(u.x * u.x + u.y * u.y));

            }
        }
    }

    for (int y = 0; y < flowx.rows; ++y)
    {
        for (int x = 0; x < flowx.cols; ++x)
        {
            Point2f u(flowx(y, x), flowy(y, x));

            if (isFlowCorrect(u))
                dst.at<Vec3b>(y, x) = computeColor(u.x / maxrad, u.y / maxrad);
        }
    }
}

static void showFlow(const char* name, const GpuMat& d_flow)
{
    GpuMat planes[2];
    cuda::split(d_flow, planes);

    Mat flowx(planes[0]);
    Mat flowy(planes[1]);

    Mat out;
    drawOpticalFlow(flowx, flowy, out);

    imshow(name, out);
}

void elementSwap(std::vector<float> &vec, int N1, int N2)
{
    float temp =vec[N1];
    vec[N1] = vec[N2];
    vec[N2] = temp;
}

float computeMedian(const std::vector<float> &arr1)
{
    vector<float> arr = arr1;
    int low, high;
    int median;
    int middle, ll, hh;
    low = 0;

    high = arr.size() - 1;
    median = (low + high) / 2;

    median = (low + high) / 2;
    for(;;)
    {
        if(high <= low)  /* One element only */
            return arr[median];

        if(high == low + 1)
        {
            /* Two elements only */
            if(arr[low] > arr[high])
                elementSwap(arr, low, high);

            return arr[median];
        }

        /* Find median of low, middle and high items; swap into position low */
        middle = (low + high) / 2;

        if(arr[middle] > arr[high])
            elementSwap(arr, middle, high);

        if(arr[low] > arr[high])
            elementSwap(arr, low, high);

        if(arr[middle] > arr[low])
            elementSwap(arr, middle, low);

        /* Swap low item (now in position middle) into position (low+1) */
        elementSwap(arr, middle, low + 1);

        /* Nibble from each end towards middle, swapping items when stuck */
        ll = low + 1;
        hh = high;

        for(;;)
        {
            do
                ll++;

            while(arr[low] > arr[ll]);

            do
                hh--;

            while(arr[hh] > arr[low]);

            if(hh < ll)
                break;

            elementSwap(arr, ll, hh);
        }

        /* Swap middle item (in position low) back into correct position */
        elementSwap(arr, low, hh);

        /* Re-set active partition */
        if(hh <= median)
            low = ll;

        if(hh >= median)
            high = hh - 1;
    }
}

float computeMedian(const Mat_<float> & input)
{
    std::vector<float> arr;
    if (input.isContinuous())
    {
        arr.assign((float*)input.datastart, (float*)input.dataend);
    }
    else
    {
        for (int i = 0; i < input.rows; ++i)
        {
            arr.insert(arr.end(), input.ptr<float>(i), input.ptr<float>(i)+input.cols);
        }
    }
    return computeMedian(arr);
}

Rect2d predictBoundingBox(const vector<Point2f> &startPoints, const vector<Point2f> &trackedPoints, Rect2d bb)
{
    Rect2d result;
    int nPts = startPoints.size();
    vector<float>ofx(nPts);
    vector<float>ofy(nPts);
    int d = 0;
    for(int i = 0; i < nPts; i++)
    {
        ofx[i] = trackedPoints[i].x - startPoints[i].x;
        ofy[i] = trackedPoints[i].y - startPoints[i].y;
    }

    float dx = computeMedian(ofx);
    float dy = computeMedian(ofy);

    int lenPdist = static_cast<int>(nPts * (nPts - 1) / 2);
    vector<float>dist0(lenPdist);
    vector<float>dist1(lenPdist);

    for(int i = 0; i < nPts; i++)
    {
        for(int j = i + 1; j < nPts; j++, d++)
        {
            dist0[d] = sqrt(pow(startPoints[i].x - startPoints[j].x, 2) + pow(startPoints[i].y - startPoints[j].y, 2));
            dist1[d] = sqrt(pow(trackedPoints[i].x - trackedPoints[j].x, 2) + pow(trackedPoints[i].y - trackedPoints[j].y, 2));
            dist0[d] = dist1[d] / dist0[d];
        }
    }

    float shift = computeMedian(dist0);

    float s0 = static_cast<float>(0.5 * (shift - 1) * bb.width);
    float s1 = static_cast<float>(0.5 * (shift - 1) * bb.height);

    result = Rect2d(Point2d(bb.tl().x - s0 + dx, bb.tl().y - s1 + dy), Point2d(bb.br().x + s0 + dx, bb.br().y + s1 + dy));
    return result;

}

Rect2d getReliablePoints(const Mat & frame0,
                         const Mat & frame1,
                         const GpuMat & forwardFlow,
                         const GpuMat & backwardFlow,
                         vector<Point2f> &startPoints,
                         vector<Point2f> &trackedPoints,
                         Rect2d inpRect = Rect2d())
{
    startPoints.clear();
    trackedPoints.clear();
    GpuMat planes[2];
    cuda::split(forwardFlow, planes);
    Mat tempx(planes[0]);
    Mat tempy(planes[1]);
    if(!inpRect.empty())
    {
        tempx = tempx(inpRect);
        tempy = tempy(inpRect);
    }

    Rect2d resultBB = Rect2d();
    Mat_<float> flowx_forward = tempx;
    Mat_<float> flowy_forward = tempy;

    cuda::split(backwardFlow, planes);
    Mat tempx1(planes[0]);
    Mat tempy1(planes[1]);

    if(!inpRect.empty())
    {
        tempx1 = tempx1(inpRect);
        tempy1 = tempy1(inpRect);
    }


    Mat_<float> flowx_backward = tempx1;
    Mat_<float> flowy_backward = tempy1;

    Size winSize(10,10);
    Mat res_im, res_template, res_result;
    Mat_<float> crossCorrelationResult = Mat_<float>(tempx.size(), (float)0);
    Mat_<float> euclidianCorrelationResult = Mat_<float>(tempx.size(), (float)0);

    // select points based on cross correlation
    for (int y = 0; y < flowx_forward.rows; ++y)
    {
        for (int x = 0; x < flowx_forward.cols; ++x)
        {
            Point2f u(flowx_forward(y, x), flowy_forward(y, x));

            if (!isFlowCorrect(u))
                continue;
            cv::getRectSubPix(frame0, winSize, Point2f(y, x), res_im);
            cv::getRectSubPix(frame1, winSize, Point2f(y + u.y, x + u.x), res_template);
            cv::matchTemplate(res_im, res_template, res_result, CV_TM_CCOEFF_NORMED);
            Mat_<float> temp = res_result;
            crossCorrelationResult(y,x) = temp(0,0);
        }
    }

    // select points based on euclidian distance
    for (int y = 0; y < flowx_backward.rows; ++y)
    {
        for (int x = 0; x < flowx_backward.cols; ++x)
        {
            Point2f u(flowx_backward(y, x), flowy_backward(y, x));
            if (!isFlowCorrect(u))
                continue;

            float im2_x = x + flowx_forward(y, x);
            float im2_y = y + flowy_forward(y, x);
            float p1 =  u.x + im2_x;
            float p2 =  u.y + im2_y;

            euclidianCorrelationResult(y,x) = sqrt((p1 - x) * (p1 - x) + (p2 - y) * (p2 - y));
        }
    }
    float medNCC = computeMedian(crossCorrelationResult);
    float medFB = computeMedian(euclidianCorrelationResult);
    for (int y = 0; y < crossCorrelationResult.rows; ++y)
    {
        for (int x = 0; x < crossCorrelationResult.cols; ++x)
        {
            if(crossCorrelationResult(y,x) >= medNCC && euclidianCorrelationResult(y,x) <= medFB)
            {
                Point2f u(flowx_forward(y, x), flowy_forward(y, x));
                startPoints.push_back(Point2f(x,y));
                trackedPoints.push_back(Point2f(x + u.x, y + u.y));
            }
        }
    }
    if(!inpRect.empty())
    {
        resultBB = predictBoundingBox(startPoints, trackedPoints, inpRect);
    }
    return resultBB;

}

std::pair<cv::Point2f, cv::Point2f> averageSubWindowFlow(const std::vector<std::pair<cv::Point2f, cv::Point2f> > & pts)
{
    cv::Point2f center;
    cv::Point2f direction;
    float x_center, y_center, x_dir, y_dir;
    x_center = 0; y_center = 0; x_dir = 0; y_dir = 0;
    if(pts.size())
    {
        int sz = pts.size();
        for (int i = 0; i < sz; i++)
        {
            x_center += pts[i].first.x;
            y_center += pts[i].first.y;
            x_dir += pts[i].second.x;
            y_dir += pts[i].second.y;
        }
        x_center /= static_cast<float>(sz);
        y_center /= static_cast<float>(sz);
        x_dir /= static_cast<float>(sz);
        y_dir /= static_cast<float>(sz);

        float div = x_dir*x_dir + y_dir*y_dir;
        if(div > 0)
        {
            x_dir  /= sqrt(x_dir*x_dir + y_dir*y_dir);
            y_dir  /= sqrt(x_dir*x_dir + y_dir*y_dir);
            return std::make_pair(cv::Point2f(x_center, y_center), cv::Point2f(x_dir, y_dir));
        }
        else
        {
            return std::make_pair(cv::Point2f(0, 0), cv::Point2f(0, 0));
        }

    }
    else
    {
        return std::make_pair(cv::Point2f(0, 0), cv::Point2f(0, 0));
    }

}


void drawPoints(Mat &drawing, const vector<Point2f> &startPoints, const vector<Point2f> &trackedPoints)
{
    int opticalFlowWindowSize = 20;
    std::vector<std::pair<cv::Point2f, cv::Point2f> > windowFlow;
    for (int y = 0; y < drawing.rows - opticalFlowWindowSize; y = y + opticalFlowWindowSize)
    {
        for (int x = 0; x < drawing.cols - opticalFlowWindowSize; x = x + opticalFlowWindowSize)
        {
            cv::Rect imgRect = cv::Rect(x, y, opticalFlowWindowSize, opticalFlowWindowSize);
            for (int i=0; i<startPoints.size(); i++)
            {
                Point center = cv::Point(startPoints[i]); // center point
                Point direction = cv::Point(trackedPoints[i].x - startPoints[i].x, trackedPoints[i].y - startPoints[i].y); // calculate direction
                //! if centre is inside the imgRect
                if(imgRect.contains(center))
                {
                    windowFlow.push_back(std::make_pair(center, direction));
                }
            }
            //cv::rectangle(rawFrame, imgRect, cv::Scalar(0,255,255), 0.5, cv::LINE_4);
            std::pair<cv::Point2f, cv::Point2f> flow = averageSubWindowFlow(windowFlow);
            int coff = static_cast<int>(10*windowFlow.size());
            coff = coff > 30 ? 30 : coff;
            cv::arrowedLine(drawing, flow.first,  flow.first + (coff*flow.second), CV_RGB(0, 0, 255), 1, 8, 0, 0.1);
            windowFlow.clear();
        }
    }
}

void image2video(const string &path, int num)
{
    std::stringstream ss;
    ss << std::setw(4) << std::setfill('0') << 1;
    std::string s = ss.str();

    // images should be in %04d format
    string filename = path + "/" + s + ".jpg";
    Mat frame = imread(filename);
    VideoWriter outputVideo;
    outputVideo.open(path + "/out.avi", CV_FOURCC('F','M','P','4'), 25 , Size(frame.cols, frame.rows), true);
    for(int i = 1 ; i <= num; i++)
    {
        cout << "writing frame - " << i << endl;
        std::stringstream ss;
        ss << std::setw(4) << std::setfill('0') << i;
        std::string s = ss.str();
        string filename = path + "/" + s + ".jpg";
        cout << filename << endl;
        Mat frame = imread(filename, IMREAD_COLOR);
        imshow("color", frame);
        waitKey(30);
        outputVideo << frame;
    }

    cout << "Finished writing" << endl;
    outputVideo.release();

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
                res.push_back(one);
        }
    }
    return res;
}

void printVector( vector<Point2f> &v)
{
    for(int i = 0; i < v.size(); i++)
    {
        cout << v[i] << endl;
    }
}

//struct comparator
//{
//    template<typename T, typename U>
//    bool operator()(T const& lhs, U const& rhs) const {
//        return lhs.x < rhs.x;
//    }
//};

int main(int argc, const char* argv[])
{

//    vector<Point2f> p1;
//    p1.push_back(Point2f(1,2));
//    p1.push_back(Point2f(2,3));
//    p1.push_back(Point2f(3,4));
//    p1.push_back(Point2f(4,5));
//    p1.push_back(Point2f(5,6));

//    vector<Point2f> p2;

//    vector<Point2f> intersection;
//    p2.push_back(Point2f(2,4));
//    p2.push_back(Point2f(2,3));
//    p2.push_back(Point2f(3,4));
//    p2.push_back(Point2f(4,5));
//    p2.push_back(Point2f(8,9));

//    std::sort (p1.begin(), p1.end(), comparator{});
//    std::sort (p2.begin(), p2.end(), comparator{});

//    std::set_intersection(p1.begin(), p1.end(),
//                          p2.begin(), p2.end(),
//                          std::back_inserter(intersection),
//                          comparator{}
//                          );

//    cout << "p1 \n";
//    printVector(p1);
//    cout << "p2 \n";
//    printVector(p2);

//    cout << "intersection \n";
//    printVector(intersection
//    return 0;

    bool paused = false;
    int m_frameH, m_frameW, m_fpsVideo, m_numFrames;
    int m_readCount = 0;
    // Initialize the optical flow algorithms
    Ptr<cuda::BroxOpticalFlow> brox = cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);
    Ptr<cuda::DensePyrLKOpticalFlow> lk = cuda::DensePyrLKOpticalFlow::create(Size(7, 7));
    Ptr<cuda::FarnebackOpticalFlow> farn = cuda::FarnebackOpticalFlow::create();
    Ptr<cuda::OpticalFlowDual_TVL1> tvl1 = cuda::OpticalFlowDual_TVL1::create();


    // take video input
    Mat currentFrame, prvFrame, currentFrameGray, prvFrameGray, drawing;
    VideoCapture vidCapture("/home/terminalx/britty/data/OR_11_Nov_17/SLAM_L/L_p.avi");
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
    vector<Rect2d> resultGT = readGT("/home/terminalx/data/Freeman1/groundtruth_rect.txt");
    Rect2d trackedRect;
    int gtCount = 0;

    namedWindow("input-video",1);
    namedWindow("drawing",1);
    vidCapture >> prvFrame;
    cvtColor(prvFrame, prvFrameGray, cv::COLOR_BGR2GRAY);

    GpuMat d_flow_forward(prvFrameGray.size(), CV_32FC2);
    GpuMat d_flow_backward(prvFrameGray.size(), CV_32FC2);
    GpuMat d_frame0f, d_frame1f;

    vector<Point2f> startPoints;
    vector<Point2f> trackedPoints;

    for(;;)
    {
        if(!paused)
        {
            m_readCount++;
            vidCapture >> currentFrame;
            currentFrame.copyTo(drawing);
            cvtColor(currentFrame, currentFrameGray, cv::COLOR_BGR2GRAY);
            if(currentFrame.empty())
                break;


            GpuMat d_frame0(prvFrameGray);
            GpuMat d_frame1(currentFrameGray);



            d_frame0.convertTo(d_frame0f, CV_32F, 1.0 / 255.0);
            d_frame1.convertTo(d_frame1f, CV_32F, 1.0 / 255.0);

            brox->calc(d_frame0f, d_frame1f, d_flow_forward);
            brox->calc(d_frame1f, d_frame0f, d_flow_backward);

            trackedRect = getReliablePoints(prvFrameGray, currentFrameGray, d_flow_forward, d_flow_backward, startPoints, trackedPoints, Rect2d()/*resultGT[gtCount]*/);
            drawPoints(drawing, startPoints, trackedPoints);
            cout << "reliable points - " << startPoints.size() << endl;


            currentFrameGray.copyTo(prvFrameGray);

            showFlow("Brox", d_flow_forward);
            //imshow("drawing", drawing);

            //rectangle(drawing, resultGT[++gtCount], Scalar(0,0,255), 3);
            //rectangle(drawing, trackedRect, Scalar(0,255,0), 2);
            imshow("drawing", drawing);

            char ch = waitKey(30);
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
