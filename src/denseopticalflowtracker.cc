#include "denseopticalflowtracker.h"

denseOpticalFlowTracker::denseOpticalFlowTracker(cv::Size sz, std::string m = "brox")
{
    method = m;
    brox = cv::cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);
    lk = cv::cuda::DensePyrLKOpticalFlow::create(cv::Size(7, 7));
    farn = cv::cuda::FarnebackOpticalFlow::create();
    tvl1 = cv::cuda::OpticalFlowDual_TVL1::create();

    //forwardFlow  = cv::cuda::GpuMat(sz, CV_32FC2);
    //backwardFlow = cv::cuda::GpuMat(sz, CV_32FC2);

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


cv::Vec3b denseOpticalFlowTracker::computeColor(float fx, float fy)
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
    static cv::Vec3i colorWheel[NCOLS];

    if (first)
    {
        int k = 0;

        for (int i = 0; i < RY; ++i, ++k)
            colorWheel[k] = cv::Vec3i(255, 255 * i / RY, 0);

        for (int i = 0; i < YG; ++i, ++k)
            colorWheel[k] = cv::Vec3i(255 - 255 * i / YG, 255, 0);

        for (int i = 0; i < GC; ++i, ++k)
            colorWheel[k] = cv::Vec3i(0, 255, 255 * i / GC);

        for (int i = 0; i < CB; ++i, ++k)
            colorWheel[k] = cv::Vec3i(0, 255 - 255 * i / CB, 255);

        for (int i = 0; i < BM; ++i, ++k)
            colorWheel[k] = cv::Vec3i(255 * i / BM, 0, 255);

        for (int i = 0; i < MR; ++i, ++k)
            colorWheel[k] = cv::Vec3i(255, 0, 255 - 255 * i / MR);

        first = false;
    }

    const float rad = sqrt(fx * fx + fy * fy);
    const float a = atan2(-fy, -fx) / (float) CV_PI;

    const float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);
    const int k0 = static_cast<int>(fk);
    const int k1 = (k0 + 1) % NCOLS;
    const float f = fk - k0;

    cv::Vec3b pix;

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

void denseOpticalFlowTracker::drawOpticalFlow(const cv::Mat_<float>& flowx, const cv::Mat_<float>& flowy, cv::Mat& dst, float maxmotion = -1)
{
    dst.create(flowx.size(), CV_8UC3);
    dst.setTo(cv::Scalar::all(0));

    // determine motion range:
    float maxrad = maxmotion;

    if (maxmotion <= 0)
    {
        maxrad = 1;
        for (int y = 0; y < flowx.rows; ++y)
        {
            for (int x = 0; x < flowx.cols; ++x)
            {
                cv::Point2f u(flowx(y, x), flowy(y, x));

                if (!isFlowCorrect(u))
                    continue;

                maxrad = std::max(maxrad, (float)sqrt(u.x * u.x + u.y * u.y));

            }
        }
    }

    for (int y = 0; y < flowx.rows; ++y)
    {
        for (int x = 0; x < flowx.cols; ++x)
        {
            cv::Point2f u(flowx(y, x), flowy(y, x));

            if (isFlowCorrect(u))
                dst.at<cv::Vec3b>(y, x) = computeColor(u.x / maxrad, u.y / maxrad);
        }
    }
}



void denseOpticalFlowTracker::showFlow(const char* name, const cv::cuda::GpuMat& d_flow)
{
    cv::cuda::GpuMat planes[2];
    cv::cuda::split(d_flow, planes);

    cv::Mat flowx(planes[0]);
    cv::Mat flowy(planes[1]);

    cv::Mat out;
    drawOpticalFlow(flowx, flowy, out);

    cv::imshow(name, out);
}


void denseOpticalFlowTracker::trackCorrespondance(const cv::Mat & frame0,
                                                  const cv::Mat & frame1,
                                                  const corr &previous_correspondance,
                                                  corr &currentFrameCorrespondance,
                                                  bool reinit,
                                                  int &totalFeatures,
                                                  cv::Rect2d inpRect,
                                                  cv::Rect2d &resultBB)
{
    int featureCount = 0;
    std::vector<cv::Point2f> resultStartPoints;
    std::vector<cv::Point2f> resultTrackedPoints;
    std::vector<int> resultUniqueID;
    std::vector<cv::Vec3b> resultColours;
    resultBB = cv::Rect2d();


    currentFrameCorrespondance.frame_1 = previous_correspondance.frame_1+1;
    currentFrameCorrespondance.frame_2 = previous_correspondance.frame_2+1;
    std::vector<cv::Point2f> startPoints = previous_correspondance.p1;
    std::vector<cv::Point2f> trackedPoints = previous_correspondance.p2;
    std::vector<int> unique_id = previous_correspondance.unique_id;
    std::vector<cv::Vec3b> colors = previous_correspondance.col;

    assert(startPoints.size() == trackedPoints.size());
    assert(trackedPoints.size() == unique_id.size());
    assert(unique_id.size() == colors.size());

    //    frame0.copyTo(prvFrame);
    //    frame1.copyTo(currentFrame);
    //prvFrame = frame0.clone();
    //currentFrame = frame1.clone();
    //if(prvFrame.channels() == 3)
    //{
    cvtColor(frame0, prvFrameGray, cv::COLOR_BGR2GRAY);
    //}

    //if(currentFrame.channels() == 3)
    //{
    cvtColor(frame1, currentFrameGray, cv::COLOR_BGR2GRAY);
    //}

    //cv::imshow("prvFrameGray", prvFrameGray);
    //cv::imshow("currentFrameGray", currentFrameGray);
    //cv::waitKey(0);


    cv::cuda::GpuMat framegpu0(prvFrameGray);
    cv::cuda::GpuMat framegpu1(currentFrameGray);

    cv::cuda::GpuMat forwardFlow(prvFrameGray.size(), CV_32FC2);
    cv::cuda::GpuMat backwardFlow(currentFrameGray.size(), CV_32FC2);

    // convert the pixel range from 0-255(uint) to 0-1(float)
    framegpu0.convertTo(framegpu0f, CV_32F, 1.0 / 255.0);
    framegpu1.convertTo(framegpu1f, CV_32F, 1.0 / 255.0);

    if(method == "brox")
    {
        tvl1->calc(framegpu0f, framegpu1f, forwardFlow);
        tvl1->calc(framegpu1f, framegpu0f, backwardFlow);
    }
    showFlow("Brox", forwardFlow);

    cv::cuda::GpuMat planes[2];
    cv::cuda::split(forwardFlow, planes);
    cv::Mat tempx(planes[0]);
    cv::Mat tempy(planes[1]);

    imshow("tempx", tempx);

    if(!inpRect.empty())
    {
        tempx = tempx(inpRect);
        tempy = tempy(inpRect);
    }
    flowx_forward = tempx;
    flowy_forward = tempy;

    //imshow("flowx_forward", flowx_forward);
    cv::cuda::split(backwardFlow, planes);
    cv::Mat tempx1(planes[0]);
    cv::Mat tempy1(planes[1]);

    if(!inpRect.empty())
    {
        tempx1 = tempx1(inpRect);
        tempy1 = tempy1(inpRect);
    }

    flowx_backward = tempx1;
    flowy_backward = tempy1;

    cv::Size winSize(5,5);
    cv::Mat res_im, res_template, res_result;

    if(startPoints.empty())
    {
        cv::Mat_<float> crossCorrelationResult = cv::Mat_<float>(tempx.size(), (float)0);
        cv::Mat_<float> euclidianCorrelationResult = cv::Mat_<float>(tempx.size(), std::numeric_limits<float>::max());
        // select points based on cross correlation
        for (int y = 3; y < flowx_forward.rows - 3; ++y)
        {
            for (int x = 3; x < flowx_forward.cols - 3; ++x)
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
        std::cout << "sum flow " << cv::sum(flowx_forward)[0] << std::endl;
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
                float val = sqrt((p1 - x) * (p1 - x) + (p2 - y) * (p2 - y));
                std::cout << "pixel val - " << val << std::endl;
                euclidianCorrelationResult(y,x) = val;
            }
        }
        float medNCC = computeMedian(crossCorrelationResult);
        float medFB = computeMedian(euclidianCorrelationResult);
        std::cout << "medFB " << medFB << std::endl;
        for (int y = 0; y < crossCorrelationResult.rows; ++y)
        {
            for (int x = 0; x < crossCorrelationResult.cols; ++x)
            {
                if(/*crossCorrelationResult(y,x) >= medNCC &&*/ euclidianCorrelationResult(y,x) <= 0.1 /*medFB*/)
                {
                    cv::Point2f u(flowx_forward(y, x), flowy_forward(y, x));
                    startPoints.push_back(cv::Point2f(x, y));
                    trackedPoints.push_back(cv::Point2f(x + u.x, y + u.y));
                    colors.push_back(frame0.at<cv::Vec3b>(cv::Point2f(x, y)));
                    unique_id.push_back(totalFeatures+featureCount++);
                }
            }
        }
        totalFeatures = unique_id.size();
    }
    else
    {
        std::vector<float> ccResult(startPoints.size(), 0);
        std::vector<float> eucResult(startPoints.size(), std::numeric_limits<float>::max());

        for(int i = 0; i < startPoints.size(); i++)
        {
            cv::Point2f u(flowx_forward(startPoints[i]), flowy_forward(startPoints[i]));
            cv::Point2f u1(flowx_backward(startPoints[i]), flowy_backward(startPoints[i]));

            if (!isFlowCorrect(u))
                continue;
            cv::getRectSubPix(frame0, winSize, cv::Point2f(startPoints[i].y, startPoints[i].x), res_im);
            cv::getRectSubPix(frame1, winSize, cv::Point2f(startPoints[i].y + u.y, startPoints[i].x + u.x), res_template);
            cv::matchTemplate(res_im, res_template, res_result, CV_TM_CCOEFF_NORMED);
            cv::Mat_<float> temp = res_result;
            ccResult[i] = temp(0,0);


            float im2_x = startPoints[i].x + flowx_forward(startPoints[i]);
            float im2_y = startPoints[i].y + flowy_forward(startPoints[i]);
            float p1 =  u1.x + im2_x;
            float p2 =  u1.y + im2_y;

            eucResult[i] = sqrt((p1 - startPoints[i].x) * (p1 - startPoints[i].x) + (p2 - startPoints[i].y) * (p2 - startPoints[i].y));
        }
        float medNCC = computeMedian(ccResult);
        float medFB = computeMedian(eucResult);

        for(int i = 0; i < startPoints.size(); i++)
        {
            if(/*ccResult[i] >= medNCC &&*/ eucResult[i] <= 0.1/*medFB*/)
            {
                cv::Point2f u(flowx_forward(startPoints[i]), flowy_forward(startPoints[i]));
                resultStartPoints.push_back(startPoints[i]);
                resultTrackedPoints.push_back(cv::Point2f(startPoints[i].x + u.x, startPoints[i].y + u.y));
                resultColours.push_back(frame0.at<cv::Vec3b>(startPoints[i]));
                resultUniqueID.push_back(unique_id[i]);
            }
        }

        startPoints = resultStartPoints;
        trackedPoints = resultTrackedPoints;
        unique_id = resultUniqueID;
        colors = resultColours;

        if(reinit)
        {
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
            //float medNCC = computeMedian(crossCorrelationResult);
            //float medFB = computeMedian(euclidianCorrelationResult);

            for (int y = 0; y < crossCorrelationResult.rows; ++y)
            {
                for (int x = 0; x < crossCorrelationResult.cols; ++x)
                {
                    if(/*crossCorrelationResult(y,x) >= medNCC &&*/ euclidianCorrelationResult(y,x) <= 0.1/*medFB*/)
                    {
                        cv::Point2f u(flowx_forward(y, x), flowy_forward(y, x));
                        startPoints.push_back(cv::Point2f(x, y));
                        trackedPoints.push_back(cv::Point2f(x + u.x, y + u.y));
                        unique_id.push_back(totalFeatures+featureCount++);
                        colors.push_back(frame0.at<cv::Vec3b>(cv::Point2f(x, y)));
                    }
                }
            }
            totalFeatures += featureCount;
        }
    }

    currentFrameCorrespondance.col = colors;
    currentFrameCorrespondance.p1 = startPoints;
    currentFrameCorrespondance.p2 = trackedPoints;
    currentFrameCorrespondance.unique_id = unique_id;


    if(!inpRect.empty())
    {
        resultBB = predictBoundingBox(startPoints, trackedPoints, inpRect);
    }
}
