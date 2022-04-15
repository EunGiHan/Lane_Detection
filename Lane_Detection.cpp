#include <iostream>
#include <vector>
#include <cmath>
#include "opencv2/opencv.hpp"
#include "Lane_Detection.h"

using namespace std;
using namespace cv;

#define LEFT 0
#define RIGHT 1

double LaneDetection::getlpos() {
    return lpos;
}


double LaneDetection::getrpos() {
    return rpos;
}


void LaneDetection::setPts() {
    /* ROI 포인트 설정하는 함수 */
    pts.push_back(Point(0, roi_offset - detectionRange));
    pts.push_back(Point(w, roi_offset - detectionRange));
    pts.push_back(Point(w, roi_offset + 20));
    pts.push_back(Point(0, roi_offset + 20));
}


vector<vector<Vec4i>> LaneDetection::SeperateLines(vector<Vec4i> lines) {
    /* 왼쪽과 오른쪽을 차선을 구분(분리)하는 함수 */

    vector<vector<Vec4i>> selected_lines;   // 이상한 직선들을 제외하고, 왼쪽/오른쪽 차선들만 저장 ({왼쪽 차선들, 오른쪽 차선들} 로 2개 요소 가짐)
    vector<Vec4i> left_lines, right_lines;  // 각각 왼쪽, 오른쪽 차선. selected_lines의 요소가 됨
    Point ini, fin; // 각각 시작점, 끝점
        
    for (Vec4i line : lines) {
        /* 직선들 기울기 계산.수평선인 것들은 제외 */

        ini = Point(line[0], line[1]);
        fin = Point(line[2], line[3]);
        double slope = static_cast<double>(fin.y - ini.y) / static_cast<double>(fin.x - ini.x + 0.00001);

        if (abs(slope) < slope_threshold)
            continue;   // 수평에 가깝게 누운 직선은 제외시킴

        if (slope > 0)
            right_lines.push_back(line);
        else
            left_lines.push_back(line);
    }

    if (left_lines.size() != 0) {
        /* 직선이 검출되었다면 추가하고, 아니면 미검출 표시 */
        selected_lines.push_back(left_lines);
        left_found = true;
    }
    else {
        left_found = false;
    }

    if (right_lines.size() != 0) {
        /* 직선이 검출되었다면 추가하고, 아니면 미검출 표시 */
        selected_lines.push_back(right_lines);
        right_found = true;
    }
    else {
        right_found = false;
    }
    
    return selected_lines;  // {왼쪽 직선들, 오른쪽 직선들} 형태로 반환
}

void LaneDetection::FindLanes(std::vector<std::vector<cv::Vec4i>> lines) {
    /* 직선들 중 대표선 하나를 만드는 함수 */

    double ini_x = 0, fin_x = 0, ini_y = 0, fin_y = 0;
    vector<Vec4i> left_lines, right_lines;

    if (left_found) {
        left_lines = lines[0];    // 함수 SperateLines()에서 도출한 왼쪽 차선들

        //if (left_lines.size() <= 0) {
        //    /* 만약을 위한 처리. 나오는 경우가 없어 주석 처리함 */
        //    left_found = false;
        //    return;
        //}
            
        for (Vec4i line : left_lines) {
            ini_x += line[0];
            ini_y += line[1];
            fin_x += line[2];
            fin_y += line[3];
        }

        ini_x = ini_x / left_lines.size();
        ini_y = ini_y / left_lines.size();
        fin_x = fin_x / left_lines.size();
        fin_y = fin_y / left_lines.size();

        left_m = (fin_y - ini_y) / (fin_x - ini_x); // 대표선의 기울기
        left_b = Point((ini_x + fin_x) / 2, (ini_y + fin_y) / 2);   // 대표선이 지나는 한 점
    }

    if (right_found) {
        ini_x = 0; fin_x = 0; ini_y = 0; fin_y = 0; // 다시 초기화

        if (left_found)
            right_lines = lines[1];    // 함수 SperateLines()에서 도출한 오른쪽 차선들
        else
            right_lines = lines[0];    // 함수 SperateLines()에서 도출한 오른쪽 차선들

        //if (right_lines.size() <= 0) {
        //    /* 만약을 위한 처리. 나오는 경우가 없어 주석 처리함 */
        //    right_found = false;
        //    return;
        //}

        for (Vec4i line : right_lines) {
            ini_x += line[0];
            ini_y += line[1];
            fin_x += line[2];
            fin_y += line[3];
        }

        ini_x = ini_x / right_lines.size();
        ini_y = ini_y / right_lines.size();
        fin_x = fin_x / right_lines.size();
        fin_y = fin_y / right_lines.size();

        right_m = (fin_y - ini_y) / (fin_x - ini_x);
        right_b = Point((ini_x + fin_x) / 2, (ini_y + fin_y) / 2);
    }
}

vector<Point> LaneDetection::FindPos() {
    /* rpos, lpos 계산 */

    vector<Point> lanes;    // 화면에 직선을 그릴 점들
    int ini_y = 420;    // 직선 시작 y좌표
    int fin_y = 380;    // 직선 끝 y좌표

    if (left_found) {
        /* offset 맞춰서 rpos & lpos 구하고, 그리기의 시작점과 끝점을 구함 */
        lpos = ((offset - left_b.y) / left_m) + left_b.x;   // offset에서 왼쪽 차선 위치
        if (lpos < 0) lpos = 0;

        double left_ini_x = ((ini_y - left_b.y) / left_m) + left_b.x;
        double left_fin_x = ((fin_y - left_b.y) / left_m) + left_b.x;
        lanes.push_back(Point(left_ini_x, ini_y));
        lanes.push_back(Point(left_fin_x, fin_y));
    }
    else
        lpos = 0;   // 미검출 시 0으로 설정

    if (right_found) {
        rpos = ((offset - right_b.y) / right_m) + right_b.x;
        if (rpos > 640) rpos = 640;

        double right_ini_x = ((ini_y - right_b.y) / right_m) + right_b.x;
        double right_fin_x = ((fin_y - right_b.y) / right_m) + right_b.x;
        lanes.push_back(Point(right_ini_x, ini_y));
        lanes.push_back(Point(right_fin_x, fin_y));
    }        
    else
        rpos = 640; // 미검출 시 640으로 설정

    if ((lpos > rpos) || (rpos - lpos < 400)) {
        /* 왼차선과 오른차선이 위치 바뀌어 있거나, 둘이 너무 가깝다면 오검출 처리 */
        lpos = 0;
        rpos = 640;
        left_found = false;
        right_found = false;
    }
    
    if ((prev_lpos != 0) && (abs(prev_lpos - lpos) > pose_gap_threshold)) {
        /* 미검출이었다가 돌아오는 상황이 아님에도, 이전 위치와 현재 위치가 차이가 크다면 오검출 처리*/
        lpos = 0;
        left_found = false;
    }
    if ((prev_rpos != 640) && (abs(prev_rpos - rpos) > pose_gap_threshold)) {
        rpos = 640;
        right_found = false;
    }

    if (left_found && prev_lpos == 0 && !lpos_queue.empty()) {
        /* 미검출이었다가 다시 검출되면 큐를 0으로 만들어 한참 전에 누적되었던 값을 초기화 */
        lpos_queue.erase(lpos_queue.begin(), lpos_queue.end());
        lpos = MAF(lpos, LEFT);
    }
    if (left_found && prev_rpos == 0 && !rpos_queue.empty()) {
        rpos_queue.erase(rpos_queue.begin(), rpos_queue.end());
        rpos = MAF(rpos, LEFT);
    }

    //cout << "prev_lpos " << prev_lpos << "\t" << "lpos " << lpos << endl;
    //cout << "prev_rpos " << prev_rpos << "\t" << "rpos " << rpos << endl;

    prev_lpos = lpos;
    prev_rpos = rpos;
    
    return lanes; // 둘 다 검출 시 {l_ini, l_fin, r_ini, r_fin} 으로 4개 Point
}

Mat LaneDetection::DrawLines(Mat src, vector<Point> lanes) {
    if (lanes.size() <= 0 || (!left_found && !right_found)) return src;
        
    if (left_found && right_found) {
        /* 둘 다 검출 시 차선 그리기 */
        line(src, lanes[0], lanes[1], Scalar(0, 0, 255), 1, LINE_AA);   // left lane
        line(src, lanes[2], lanes[3], Scalar(0, 0, 255), 1, LINE_AA);   // right lane
    }
    else if ((!left_found && right_found) || (left_found && !right_found)) {
        /* 하나만 검출 시 차선 그리기 */
        line(src, lanes[0], lanes[1], Scalar(0, 0, 255), 1, LINE_AA);
    }

    rectangle(src, Point(lpos - box_width / 2, offset - box_height / 2), Point(lpos + box_width / 2, offset + box_height / 2), Scalar(255, 255, 0), 2); // 왼차선
    rectangle(src, Point(rpos - box_width / 2, offset - box_height / 2), Point(rpos + box_width / 2, offset + box_height / 2), Scalar(255, 0, 255), 2); // 오른차선

    return src;
}

double LaneDetection::MAF(double pos, int left_right) {
    /* 이동평균필터로 lpos, rpos 계산 */

    double sum = 0; // 큐 내부 원소 합 구할 변수
    double weights_sum = 0; // 가중치 합 구할 변수

    if (left_right == LEFT) {
        if (lpos_queue.size() >= 5)
            lpos_queue.erase(lpos_queue.begin());
        lpos_queue.push_back(pos);

        for (int i = 0; i < lpos_queue.size(); i++) {
            sum += (lpos_queue[i] * maf_weight[i]);
            weights_sum += maf_weight[i];
        }
    }
    else {
        if (rpos_queue.size() >= 5)
            rpos_queue.erase(rpos_queue.begin());
        rpos_queue.push_back(pos);

        for (int i = 0; i < rpos_queue.size(); i++) {
            sum += (rpos_queue[i] * maf_weight[i]);
            weights_sum += maf_weight[i];
        }
    }

    return (sum / weights_sum);
}