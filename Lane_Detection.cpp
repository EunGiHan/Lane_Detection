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
    /* ROI ����Ʈ �����ϴ� �Լ� */
    pts.push_back(Point(0, roi_offset - detectionRange));
    pts.push_back(Point(w, roi_offset - detectionRange));
    pts.push_back(Point(w, roi_offset + 20));
    pts.push_back(Point(0, roi_offset + 20));
}


vector<vector<Vec4i>> LaneDetection::SeperateLines(vector<Vec4i> lines) {
    /* ���ʰ� �������� ������ ����(�и�)�ϴ� �Լ� */

    vector<vector<Vec4i>> selected_lines;   // �̻��� �������� �����ϰ�, ����/������ �����鸸 ���� ({���� ������, ������ ������} �� 2�� ��� ����)
    vector<Vec4i> left_lines, right_lines;  // ���� ����, ������ ����. selected_lines�� ��Ұ� ��
    Point ini, fin; // ���� ������, ����
        
    for (Vec4i line : lines) {
        /* ������ ���� ���.������ �͵��� ���� */

        ini = Point(line[0], line[1]);
        fin = Point(line[2], line[3]);
        double slope = static_cast<double>(fin.y - ini.y) / static_cast<double>(fin.x - ini.x + 0.00001);

        if (abs(slope) < slope_threshold)
            continue;   // ���� ������ ���� ������ ���ܽ�Ŵ

        if (slope > 0)
            right_lines.push_back(line);
        else
            left_lines.push_back(line);
    }

    if (left_lines.size() != 0) {
        /* ������ ����Ǿ��ٸ� �߰��ϰ�, �ƴϸ� �̰��� ǥ�� */
        selected_lines.push_back(left_lines);
        left_found = true;
    }
    else {
        left_found = false;
    }

    if (right_lines.size() != 0) {
        /* ������ ����Ǿ��ٸ� �߰��ϰ�, �ƴϸ� �̰��� ǥ�� */
        selected_lines.push_back(right_lines);
        right_found = true;
    }
    else {
        right_found = false;
    }
    
    return selected_lines;  // {���� ������, ������ ������} ���·� ��ȯ
}

void LaneDetection::FindLanes(std::vector<std::vector<cv::Vec4i>> lines) {
    /* ������ �� ��ǥ�� �ϳ��� ����� �Լ� */

    double ini_x = 0, fin_x = 0, ini_y = 0, fin_y = 0;
    vector<Vec4i> left_lines, right_lines;

    if (left_found) {
        left_lines = lines[0];    // �Լ� SperateLines()���� ������ ���� ������

        //if (left_lines.size() <= 0) {
        //    /* ������ ���� ó��. ������ ��찡 ���� �ּ� ó���� */
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

        left_m = (fin_y - ini_y) / (fin_x - ini_x); // ��ǥ���� ����
        left_b = Point((ini_x + fin_x) / 2, (ini_y + fin_y) / 2);   // ��ǥ���� ������ �� ��
    }

    if (right_found) {
        ini_x = 0; fin_x = 0; ini_y = 0; fin_y = 0; // �ٽ� �ʱ�ȭ

        if (left_found)
            right_lines = lines[1];    // �Լ� SperateLines()���� ������ ������ ������
        else
            right_lines = lines[0];    // �Լ� SperateLines()���� ������ ������ ������

        //if (right_lines.size() <= 0) {
        //    /* ������ ���� ó��. ������ ��찡 ���� �ּ� ó���� */
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
    /* rpos, lpos ��� */

    vector<Point> lanes;    // ȭ�鿡 ������ �׸� ����
    int ini_y = 420;    // ���� ���� y��ǥ
    int fin_y = 380;    // ���� �� y��ǥ

    if (left_found) {
        /* offset ���缭 rpos & lpos ���ϰ�, �׸����� �������� ������ ���� */
        lpos = ((offset - left_b.y) / left_m) + left_b.x;   // offset���� ���� ���� ��ġ
        if (lpos < 0) lpos = 0;

        double left_ini_x = ((ini_y - left_b.y) / left_m) + left_b.x;
        double left_fin_x = ((fin_y - left_b.y) / left_m) + left_b.x;
        lanes.push_back(Point(left_ini_x, ini_y));
        lanes.push_back(Point(left_fin_x, fin_y));
    }
    else
        lpos = 0;   // �̰��� �� 0���� ����

    if (right_found) {
        rpos = ((offset - right_b.y) / right_m) + right_b.x;
        if (rpos > 640) rpos = 640;

        double right_ini_x = ((ini_y - right_b.y) / right_m) + right_b.x;
        double right_fin_x = ((fin_y - right_b.y) / right_m) + right_b.x;
        lanes.push_back(Point(right_ini_x, ini_y));
        lanes.push_back(Point(right_fin_x, fin_y));
    }        
    else
        rpos = 640; // �̰��� �� 640���� ����

    if ((lpos > rpos) || (rpos - lpos < 400)) {
        /* �������� ���������� ��ġ �ٲ�� �ְų�, ���� �ʹ� �����ٸ� ������ ó�� */
        lpos = 0;
        rpos = 640;
        left_found = false;
        right_found = false;
    }
    
    if ((prev_lpos != 0) && (abs(prev_lpos - lpos) > pose_gap_threshold)) {
        /* �̰����̾��ٰ� ���ƿ��� ��Ȳ�� �ƴԿ���, ���� ��ġ�� ���� ��ġ�� ���̰� ũ�ٸ� ������ ó��*/
        lpos = 0;
        left_found = false;
    }
    if ((prev_rpos != 640) && (abs(prev_rpos - rpos) > pose_gap_threshold)) {
        rpos = 640;
        right_found = false;
    }

    if (left_found && prev_lpos == 0 && !lpos_queue.empty()) {
        /* �̰����̾��ٰ� �ٽ� ����Ǹ� ť�� 0���� ����� ���� ���� �����Ǿ��� ���� �ʱ�ȭ */
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
    
    return lanes; // �� �� ���� �� {l_ini, l_fin, r_ini, r_fin} ���� 4�� Point
}

Mat LaneDetection::DrawLines(Mat src, vector<Point> lanes) {
    if (lanes.size() <= 0 || (!left_found && !right_found)) return src;
        
    if (left_found && right_found) {
        /* �� �� ���� �� ���� �׸��� */
        line(src, lanes[0], lanes[1], Scalar(0, 0, 255), 1, LINE_AA);   // left lane
        line(src, lanes[2], lanes[3], Scalar(0, 0, 255), 1, LINE_AA);   // right lane
    }
    else if ((!left_found && right_found) || (left_found && !right_found)) {
        /* �ϳ��� ���� �� ���� �׸��� */
        line(src, lanes[0], lanes[1], Scalar(0, 0, 255), 1, LINE_AA);
    }

    rectangle(src, Point(lpos - box_width / 2, offset - box_height / 2), Point(lpos + box_width / 2, offset + box_height / 2), Scalar(255, 255, 0), 2); // ������
    rectangle(src, Point(rpos - box_width / 2, offset - box_height / 2), Point(rpos + box_width / 2, offset + box_height / 2), Scalar(255, 0, 255), 2); // ��������

    return src;
}

double LaneDetection::MAF(double pos, int left_right) {
    /* �̵�������ͷ� lpos, rpos ��� */

    double sum = 0; // ť ���� ���� �� ���� ����
    double weights_sum = 0; // ����ġ �� ���� ����

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