#pragma once
#include <iostream>
#include <vector>
#include <queue>
#include "opencv2/opencv.hpp"

class LaneDetection
{
private:
    int offset = 400;   // 차선의 위치를 찾을 offset
    int roi_offset = 380;   // 차선을 검출할 offset. 400으로 설정 시 차체와 ROI가 겹쳐 오검출이 많아, 따로 설정
    int detectionRange = 10;    // ROI 높이 절반
    double slope_threshold = 0.2;   // 기울기의 (절댓값) 최솟값
    double pose_gap_threshold = 60; // 이전 차선 위치와 현재 차선 위치의 최소 격차

    double left_m, right_m;     // 왼쪽과 오른쪽 차선의 기울기
    cv::Point left_b, right_b;  // 왼쪽과 오른쪽 직선이 지나는 한 점
    
    int lpos = 0;
    int rpos = 640;
    double prev_lpos = lpos;    // 직전 lpos
    double prev_rpos = rpos;    // 직전 rpos

    std::vector<double> lpos_queue; // lpos 이동평균필터
    std::vector<double> rpos_queue; // rpos 이동평균필터
    std::vector<double> maf_weight = { 1, 2, 3, 4, 5 }; // 이동평균필터에 쓰일 weight

    int box_width = 20;     // 차선 위치를 그릴 box의 가로 크기
    int box_height = 20;    // 차선 위치를 그릴 box의 세로 크기

public:
    int h, w;   // 프레임의 세로, 가로 크기
    bool left_found = false;    // 왼쪽 차선을 찾았는지 여부
    bool right_found = false;   // 오쪽 차선을 찾았는지 여부
    std::vector<cv::Point> pts; // ROI를 설정할 포인트들

    double getlpos();
    double getrpos();
    void setPts();  // ROI 포인트 설정
    std::vector<std::vector<cv::Vec4i>> SeperateLines(std::vector<cv::Vec4i> lines);    // 왼쪽, 오른쪽 차선 분리
    void FindLanes(std::vector<std::vector<cv::Vec4i>> lines);  // 각 방향(왼, 오)의 차선들을 각 방향별로 하나의 직선으로 통합
    std::vector<cv::Point> FindPos();   // rpos, lpos 계산
    double MAF(double pos, int left_right); // 이동평균필터로 lpos, rpos 계산
    cv::Mat DrawLines(cv::Mat src, std::vector<cv::Point> lanes);   // 화면에 차선, 위치 그리기
};