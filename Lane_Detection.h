#pragma once
#include <iostream>
#include <vector>
#include <queue>
#include "opencv2/opencv.hpp"

class LaneDetection
{
private:
    int offset = 400;   // ������ ��ġ�� ã�� offset
    int roi_offset = 380;   // ������ ������ offset. 400���� ���� �� ��ü�� ROI�� ���� �������� ����, ���� ����
    int detectionRange = 10;    // ROI ���� ����
    double slope_threshold = 0.2;   // ������ (����) �ּڰ�
    double pose_gap_threshold = 60; // ���� ���� ��ġ�� ���� ���� ��ġ�� �ּ� ����

    double left_m, right_m;     // ���ʰ� ������ ������ ����
    cv::Point left_b, right_b;  // ���ʰ� ������ ������ ������ �� ��
    
    int lpos = 0;
    int rpos = 640;
    double prev_lpos = lpos;    // ���� lpos
    double prev_rpos = rpos;    // ���� rpos

    std::vector<double> lpos_queue; // lpos �̵��������
    std::vector<double> rpos_queue; // rpos �̵��������
    std::vector<double> maf_weight = { 1, 2, 3, 4, 5 }; // �̵�������Ϳ� ���� weight

    int box_width = 20;     // ���� ��ġ�� �׸� box�� ���� ũ��
    int box_height = 20;    // ���� ��ġ�� �׸� box�� ���� ũ��

public:
    int h, w;   // �������� ����, ���� ũ��
    bool left_found = false;    // ���� ������ ã�Ҵ��� ����
    bool right_found = false;   // ���� ������ ã�Ҵ��� ����
    std::vector<cv::Point> pts; // ROI�� ������ ����Ʈ��

    double getlpos();
    double getrpos();
    void setPts();  // ROI ����Ʈ ����
    std::vector<std::vector<cv::Vec4i>> SeperateLines(std::vector<cv::Vec4i> lines);    // ����, ������ ���� �и�
    void FindLanes(std::vector<std::vector<cv::Vec4i>> lines);  // �� ����(��, ��)�� �������� �� ���⺰�� �ϳ��� �������� ����
    std::vector<cv::Point> FindPos();   // rpos, lpos ���
    double MAF(double pos, int left_right); // �̵�������ͷ� lpos, rpos ���
    cv::Mat DrawLines(cv::Mat src, std::vector<cv::Point> lanes);   // ȭ�鿡 ����, ��ġ �׸���
};