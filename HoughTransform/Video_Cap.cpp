#include <iostream>
#include <algorithm>
#include <cmath>
#include "opencv2/opencv.hpp"
#include "Video_Cap.h"

constexpr auto HLS_CHANNEL = 1;
constexpr auto LAB_CHANNEL = 2;

using namespace std;
using namespace cv;

Mat VideoCap::binaryImg(Mat frame) {
    Mat gray, a, b, c;
    GaussianBlur(frame, frame, Point(5, 5), 1, 1, 4);
    a = normalize_HLS_L(frame);
    b = normalize_LAB_L(frame);
    bitwise_and(a, b, c);
    //return c;
    return a;
}

Mat VideoCap::make_zeros(Mat img)
{
    return Mat::zeros(img.rows, img.cols, img.type());
}

Mat VideoCap::make_ones(Mat img)
{
    return Mat::ones(img.rows, img.cols, img.type());
}

Mat VideoCap::normalize_HLS_L(Mat unWarp)
{
    /* normalizing L color channel pixel from hls img. */
    Mat imgHLS_L, imgNormal;
    double minVal, maxVal;
    Point minLoc, maxLoc;
    int lowThres = 150;

    // get a single channel img(filtered one.)
    imgHLS_L = filterImg(unWarp, HLS_CHANNEL, 1);

    // get max, min value of the matrix.
    minMaxLoc(imgHLS_L, &minVal, &maxVal, &minLoc, &maxLoc);

    // make normalized img.
    imgNormal = (255 / maxVal) * imgHLS_L;

    // apply threshold for L channel.
    Mat imgOut = make_zeros(imgNormal);
    adaptiveThreshold(imgNormal, imgOut, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 27, 21);
    medianBlur(imgOut, imgOut, 5);
    Mat element5(3, 3, CV_8U, cv::Scalar(1));
    morphologyEx(imgOut, imgOut, MORPH_OPEN, element5);
    morphologyEx(imgOut, imgOut, MORPH_CLOSE, element5);
    return imgOut;
}

Mat VideoCap::normalize_LAB_L(Mat unWarp)
{
    /* normalizing B color channel pixel from LAB img. */
    Mat imgLAB_L, imgNormal;
    double minVal, maxVal;
    Point minLoc, maxLoc;
    int yellowCrit = 175;
    int lowThres = 130;

    // get a single channel img(filtered one.)
    imgLAB_L = filterImg(unWarp, LAB_CHANNEL, 0);
    // get max, min value of the matrix.
    minMaxLoc(imgLAB_L, &minVal, &maxVal, &minLoc, &maxLoc);

    // (conditional) make normalized img.
    // B channel means a range from blue(0) to yellow(255).
    // So, the bigger values, it becomes close to yellow color.(yellow lane)
    if (maxVal > yellowCrit)
    {
        imgNormal = (255 / maxVal) * imgLAB_L;
    }
    else
    {
        imgNormal = imgLAB_L;
    }

    // apply threshold for L channel.
    Mat imgOut = Mat::zeros(imgNormal.rows, imgNormal.cols, imgNormal.type());
    //imshow("imgNormal", imgNormal);
    adaptiveThreshold(imgNormal, imgOut, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 15, 2);
    medianBlur(imgOut, imgOut, 5);
    Mat element5(3, 3, CV_8U, cv::Scalar(1));
    morphologyEx(imgOut, imgOut, MORPH_OPEN, element5);
    morphologyEx(imgOut, imgOut, MORPH_CLOSE, element5);
    return imgOut;
}

Mat VideoCap::grayTo_Dir(Mat gray, int dirKernelSize, double dir_threshold[])
{
    Mat sobelX, sobelY;
    Mat min_dir, max_dir;
    Mat binaryOutput;

    // sobel edge both x and y direction
    Sobel(gray, sobelX, CV_64F, 1, 0, dirKernelSize, 1, 0, BORDER_DEFAULT);
    Sobel(gray, sobelY, CV_64F, 0, 1, dirKernelSize, 1, 0, BORDER_DEFAULT);
    sobelX = abs(sobelX);
    sobelY = abs(sobelY);
    sobel = sobelX + sobelY;
    // GET gradient direction by calculating arctan value for absoute ones.
    Mat gradDir = Mat::ones(sobelX.rows, sobelX.cols, sobelX.type());
    for (int i = 0; i < sobelX.rows; i++)
    {
        for (int j = 0; j < sobelX.cols; j++)
        {
            double gradRadian = 0.0;
            gradRadian = atan2(
                sobelY.at<double>(i, j),
                sobelX.at<double>(i, j));
            if (gradRadian > dir_threshold[0] && gradRadian < dir_threshold[1])
            {
                gradDir.at<double>(i, j) = 0.0;
            }
        }
    }
    convertScaleAbs(gradDir, gradDir);
    return gradDir;
}

Mat VideoCap::filterImg(Mat imgUnwarp, int toColorChannel, int mode)
{
    /*
    channel mode definition.
        0 : Hue
        1 : Lightness
        2 : Saturation

        hue max : 179, l and s max : 255
    */
    Mat imgConverted;
    int height = imgUnwarp.cols;
    int width = imgUnwarp.rows;
    Mat imgOUT = Mat::zeros(width, height, CV_8UC1);

    /* 1. convert color channel from BGR to HLS or LAB. */
    if (toColorChannel == HLS_CHANNEL)
    {
        cvtColor(imgUnwarp, imgConverted, COLOR_BGR2HLS);
    }
    else if (toColorChannel == LAB_CHANNEL)
    {
        cvtColor(imgUnwarp, imgConverted, COLOR_BGR2Lab);
    }

    uint8_t* pixelPtr = (uint8_t*)imgConverted.data;
    int cn = imgConverted.channels();

    switch (mode)
    {
    case 0:
        // set H space Only  // set L space Only
        for (int i = 0; i < imgConverted.rows; i++)
        {
            for (int j = 0; j < imgConverted.cols; j++)
            {
                imgOUT.at<uint8_t>(i, j) = pixelPtr[i * imgConverted.cols * cn + j * cn + 0];
            }
        }
        break;
    case 1:
        // set L space Only  // set A space Only
        for (int i = 0; i < imgConverted.rows; i++)
        {
            for (int j = 0; j < imgConverted.cols; j++)
            {
                imgOUT.at<uint8_t>(i, j) = pixelPtr[i * imgConverted.cols * cn + j * cn + 1];
            }
        }
        break;

    case 2:
        // set S space Only  // set B space Only
        for (int i = 0; i < imgConverted.rows; i++)
        {
            for (int j = 0; j < imgConverted.cols; j++)
            {
                imgOUT.at<uint8_t>(i, j) = pixelPtr[i * imgConverted.cols * cn + j * cn + 2];
            }
        }
        break;

    default:
        break;
    }
    return imgOUT;
}
