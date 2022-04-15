#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "Lane_Detection.h"
#include "Video_Cap.h"

using namespace std;
using namespace cv;

int main()
{
    VideoCap vc;
    LaneDetection ld;

    Mat frame, bin, roi, edge, edge_roi;
    vector<Vec4i> lines;    // �������� ��ȯ ����� ���� ������
    vector<vector<Vec4i>> selected_lines;   // ���� ����� ������ �����ϰ� ���͸��� ������
    vector<Point> lanes;    // ȭ�鿡 �׸� ������ ����Ʈ�� ����

    int frame_cnt = 0;  // 30�������� �� ����
    
    ofstream output("output.csv");   // Save 2D array to CSV
    VideoCapture cap("subProject.avi");

    if (!cap.isOpened()) {
        cerr << "Video open failed!" << endl;
        return -1;
    }

    ld.h = cvRound(cap.get(CAP_PROP_FRAME_HEIGHT));
    ld.w = cvRound(cap.get(CAP_PROP_FRAME_WIDTH));
    
    Mat mask = Mat::zeros(ld.h, ld.w, CV_8UC1);
    ld.setPts();    // ROI ����
    fillPoly(mask, ld.pts, Scalar(255));

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            cout << "Empty Frame!" << endl;
            break;
        }

        bin = vc.binaryImg(frame);  // ���� �̹����� ��ȯ
        Canny(bin, edge, 120, 200); // ���� ����
        edge.copyTo(roi, mask); // ROI ��ŭ �ڸ���
        HoughLinesP(roi, lines, 1, CV_PI / 180, 20, 5, 10); // ���� ����


        if (lines.size() > 0) {
            selected_lines = ld.SeperateLines(lines);   // ���� ������, ���� �������� ������

            /* 
            // ROI ȭ�鿡�� �� �������� Ȯ���ϱ� ���� �κ�. �ӵ� ����� ���� �׽�Ʈ �ܿ� �ּ� ó��
            cvtColor(roi, roi, COLOR_GRAY2BGR);
            for (vector<Vec4i> lines : selected_lines) {
                for (Vec4i l : lines) {
                    line(roi, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 255, 0), 1, LINE_AA);
                }
            }*/

            ld.FindLanes(selected_lines);   // �� ������ �� ��ǥ ����(����)�� ������
            lanes = ld.FindPos();   // rpos, lpos�� ����

            /*
            // ȭ�鿡 ������ rpos&lpos�� �׸��� �κ�. �ӵ� ����� ���� �׽�Ʈ �ܿ� �ּ� ó��
            frame = ld.DrawLines(frame, lanes);
            roi = ld.DrawLines(roi, lanes);
            */

            if (frame_cnt == 30) {
                output << ld.getlpos() << ", " << ld.getrpos() << endl;
                frame_cnt = 0;
            }
            else {
                frame_cnt++;
            }
            
        }
        else
        {
            output << 0 << ", " << 640 << endl;
        }

        /*
        // ȭ�鿡 ����� �׸��� �κ�. �ӵ� ����� ���� �׽�Ʈ �ܿ� �ּ� ó��
        imshow("roi", roi);
        imshow("frame", frame);*/

        waitKey(1);
    }

    output.close();
    cap.release();
    destroyAllWindows();
}
