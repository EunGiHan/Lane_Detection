#pragma once
#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"

class VideoCap {
private:
    cv::VideoCapture cap = cv::VideoCapture("subProject.avi");
    cv::Mat frame, hls, lab, sobel, dir, mag;

public:
    //int open();
    //void play();
    cv::Mat binaryImg(cv::Mat frame);
    cv::Mat make_zeros(cv::Mat img);
    cv::Mat make_ones(cv::Mat img);
    cv::Mat normalize_HLS_L(cv::Mat unWarp);
    cv::Mat normalize_LAB_L(cv::Mat unWarp);
    cv::Mat grayTo_Dir(cv::Mat gray, int dirKernelSize, double dir_threshold[]);
    cv::Mat filterImg(cv::Mat imgUnwarp, int toColorChannel, int mode);
};
