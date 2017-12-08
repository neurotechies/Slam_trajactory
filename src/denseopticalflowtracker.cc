#include "denseopticalflowtracker.h"

denseOpticalFlowTracker::denseOpticalFlowTracker(cv::Size sz, std::string m = "brox")
{
    method = m;
    brox = cv::cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);
    lk = cv::cuda::DensePyrLKOpticalFlow::create(cv::Size(7, 7));
    farn = cv::cuda::FarnebackOpticalFlow::create();
    tvl1 = cv::cuda::OpticalFlowDual_TVL1::create();

    d_flow_forward  = cv::cuda::GpuMat(sz, CV_32FC2);
    d_flow_backward = cv::cuda::GpuMat(sz, CV_32FC2);

}


inline bool denseOpticalFlowTracker::isFlowCorrect(cv::Point2f u)
{
    return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9;
}


void denseOpticalFlowTracker::elementSwap(std::vector<float> &vec, int N1, int N2)
{
    float temp =vec[N1];
    vec[N1] = vec[N2];
    vec[N2] = temp;
}

float denseOpticalFlowTracker::computeMedian(const std::vector<float> &arr1)
{
    std::vector<float> arr = arr1;
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


float denseOpticalFlowTracker::computeMedian(const cv::Mat_<float> & input)
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


cv::Rect2d denseOpticalFlowTracker::predictBoundingBox(const std::vector<cv::Point2f> &startPoints,
                                                       const std::vector<cv::Point2f> &trackedPoints,
                                                       cv::Rect2d bb)
{
    cv::Rect2d result;
    int nPts = startPoints.size();
    std::vector<float>ofx(nPts);
    std::vector<float>ofy(nPts);
    int d = 0;
    for(int i = 0; i < nPts; i++)
    {
        ofx[i] = trackedPoints[i].x - startPoints[i].x;
        ofy[i] = trackedPoints[i].y - startPoints[i].y;
    }

    float dx = computeMedian(ofx);
    float dy = computeMedian(ofy);

    int lenPdist = static_cast<int>(nPts * (nPts - 1) / 2);
    std::vector<float>dist0(lenPdist);
    std::vector<float>dist1(lenPdist);

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

    result = cv::Rect2d(cv::Point2d(bb.tl().x - s0 + dx, bb.tl().y - s1 + dy), cv::Point2d(bb.br().x + s0 + dx, bb.br().y + s1 + dy));
    return result;

}


cv::Rect2d denseOpticalFlowTracker::getReliablePoints(const cv::Mat & frame0,
                         const cv::Mat & frame1,
                         const cv::cuda::GpuMat & forwardFlow,
                         const cv::cuda::GpuMat & backwardFlow,
                         std::vector<cv::Point2f> &startPoints,
                         std::vector<cv::Point2f> &trackedPoints,
                         cv::Mat_<float> &flowx_forward,
                         cv::Mat_<float> &flowy_forward,
                         cv::Rect2d inpRect = cv::Rect2d())
{
    startPoints.clear();
    trackedPoints.clear();
    cv::cuda::GpuMat planes[2];
    cv::cuda::split(forwardFlow, planes);
    cv::Mat tempx(planes[0]);
    cv::Mat tempy(planes[1]);
    if(!inpRect.empty())
    {
        tempx = tempx(inpRect);
        tempy = tempy(inpRect);
    }

    cv::Rect2d resultBB = cv::Rect2d();
    flowx_forward = tempx;
    flowy_forward = tempy;

    cv::cuda::split(backwardFlow, planes);
    cv::Mat tempx1(planes[0]);
    cv::Mat tempy1(planes[1]);

    if(!inpRect.empty())
    {
        tempx1 = tempx1(inpRect);
        tempy1 = tempy1(inpRect);
    }


    cv::Mat_<float> flowx_backward = tempx1;
    cv::Mat_<float> flowy_backward = tempy1;

    cv::Size winSize(10,10);
    cv::Mat res_im, res_template, res_result;
    cv::Mat_<float> crossCorrelationResult = cv::Mat_<float>(tempx.size(), (float)0);
    cv::Mat_<float> euclidianCorrelationResult = cv::Mat_<float>(tempx.size(), (float)0);

    // select points based on cross correlation
    for (int y = 0; y < flowx_forward.rows; ++y)
    {
        for (int x = 0; x < flowx_forward.cols; ++x)
        {
            cv::Point2f u(flowx_forward(y, x), flowy_forward(y, x));

            if (!isFlowCorrect(u))
                continue;
            cv::getRectSubPix(frame0, winSize, cv::Point2f(y, x), res_im);
            cv::getRectSubPix(frame1, winSize, cv::Point2f(y + u.y, x + u.x), res_template);
            cv::matchTemplate(res_im, res_template, res_result, CV_TM_CCOEFF_NORMED);
            cv::Mat_<float> temp = res_result;
            crossCorrelationResult(y,x) = temp(0,0);
        }
    }

    // select points based on euclidian distance
    for (int y = 0; y < flowx_backward.rows; ++y)
    {
        for (int x = 0; x < flowx_backward.cols; ++x)
        {
            cv::Point2f u(flowx_backward(y, x), flowy_backward(y, x));
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
                cv::Point2f u(flowx_forward(y, x), flowy_forward(y, x));
                startPoints.push_back(cv::Point2f(x,y));
                trackedPoints.push_back(cv::Point2f(x + u.x, y + u.y));
            }
        }
    }
    if(!inpRect.empty())
    {
        resultBB = predictBoundingBox(startPoints, trackedPoints, inpRect);
    }
    return resultBB;


}


void denseOpticalFlowTracker::trackPoints(const cv::Mat &f_prv,
                 const cv::Mat &f_curr,
                 std::vector<cv::Point2f> &prvFramePoints,
                 std::vector<cv::Point2f> &currentFramePoints,
                 bool reinit = false)
{
    f_prv.copyTo(prvFrame);
    f_curr.copyTo(currentFrame);
    if(prvFrame.channels() == 3)
    {
        cvtColor(prvFrame, prvFrame, cv::COLOR_BGR2GRAY);
    }

    if(currentFrame.channels() == 3)
    {
        cvtColor(currentFrame, currentFrame, cv::COLOR_BGR2GRAY);
    }

    cv::cuda::GpuMat frame0(prvFrame);
    cv::cuda::GpuMat frame1(currentFrame);

    // convert the pixel range from 0-255(uint) to 0-1(float)
    frame0.convertTo(frame0, CV_32F, 1.0 / 255.0);
    frame1.convertTo(frame1, CV_32F, 1.0 / 255.0);

    if(method == "brox")
    {
        brox->calc(frame0, frame1, d_flow_forward);
        brox->calc(frame1, frame0, d_flow_backward);

        if(prvFramePoints.empty() || reinit)
        {
            // compute the dense optical flow on the whole image
            if(prvFramePoints.empty())
            {
                prvFramePoints.clear();
                currentFramePoints.clear();
                cv::Mat_<float> flowx_forward, flowy_forward;
                getReliablePoints(prvFrame,
                                  currentFrame,
                                  d_flow_forward,
                                  d_flow_backward,
                                  prvFramePoints,
                                  currentFramePoints,
                                  flowx_forward,
                                  flowy_forward);
            }
            else if(reinit)
            {
                std::vector<cv::Point2f> tempPrevious = prvFramePoints;
                std::vector<cv::Point2f> tempCurrent  =  currentFramePoints;
                cv::Mat_<float> flowx_forward, flowy_forward;
                getReliablePoints(prvFrame,
                                  currentFrame,
                                  d_flow_forward,
                                  d_flow_backward,
                                  prvFramePoints,
                                  currentFramePoints,
                                  flowx_forward,
                                  flowy_forward);
                tempPrevious.insert(tempPrevious.end(), prvFramePoints.begin(), prvFramePoints.end());
                tempCurrent.insert(tempCurrent.end(), currentFramePoints.begin(), currentFramePoints.end());

                prvFramePoints = tempPrevious;
                currentFramePoints = tempCurrent;
            }
        }
        else
        {
            // track the points in the next frame
            std::vector<cv::Point2f> tempPrevious;
            std::vector<cv::Point2f> tempCurrent;
            std::vector<cv::Point2f> resultPreviousPoints;

            getReliablePoints(prvFrame,
                              currentFrame,
                              d_flow_forward,
                              d_flow_backward,
                              tempPrevious,
                              tempCurrent,
                              flowx_forward,
                              flowy_forward);
            // find the intersection of prvFramePoints with the tempPrevious
            std::sort (tempPrevious.begin(), tempPrevious.end(), comparator_cvPoint2f{});
            std::sort (prvFramePoints.begin(), prvFramePoints.end(), comparator_cvPoint2f{});

            std::set_intersection(tempPrevious.begin(), tempPrevious.end(),
                                  prvFramePoints.begin(), prvFramePoints.end(),
                                  std::back_inserter(resultPreviousPoints),
                                  comparator_cvPoint2f{});
            prvFramePoints.clear();
            currentFramePoints.clear();
            for(int i = 0; i < resultPreviousPoints.size(); i++)
            {
                cv::Point2f pt = resultPreviousPoints[i];
                cv::Point2f u(flowx_forward(pt), flowy_forward(pt));
                prvFramePoints.push_back(cv::Point2f(pt.x, pt.y));
                currentFramePoints.push_back(cv::Point2f(pt.x + u.x, pt.y + u.y));
            }
        }
    }

}

void denseOpticalFlowTracker::trackPoints(const cv::Mat &prvFrame,
                 const cv::Mat &currentFrame,
                 corr &prev_correspondance,
                 corr &current_correspondance,
                 bool reinit)
{

}
