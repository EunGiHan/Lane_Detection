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
    vector<Vec4i> lines;    // 허프라인 변환 결과를 담을 직선들
    vector<vector<Vec4i>> selected_lines;   // 수평에 가까운 직선은 제외하고 필터링된 직선들
    vector<Point> lanes;    // 화면에 그릴 직선의 포인트들 모음

    int frame_cnt = 0;  // 30프레임을 셀 변수
    
    ofstream output("output.csv");   // Save 2D array to CSV
    VideoCapture cap("subProject.avi");

    if (!cap.isOpened()) {
        cerr << "Video open failed!" << endl;
        return -1;
    }

    ld.h = cvRound(cap.get(CAP_PROP_FRAME_HEIGHT));
    ld.w = cvRound(cap.get(CAP_PROP_FRAME_WIDTH));
    
    Mat mask = Mat::zeros(ld.h, ld.w, CV_8UC1);
    ld.setPts();    // ROI 설정
    fillPoly(mask, ld.pts, Scalar(255));

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            cout << "Empty Frame!" << endl;
            break;
        }

        bin = vc.binaryImg(frame);  // 이진 이미지로 변환
        Canny(bin, edge, 120, 200); // 에지 검출
        edge.copyTo(roi, mask); // ROI 만큼 자르기
        HoughLinesP(roi, lines, 1, CV_PI / 180, 20, 5, 10); // 직선 검출


        if (lines.size() > 0) {
            selected_lines = ld.SeperateLines(lines);   // 좌측 직선들, 우측 직선들을 구분함

            /* 
            // ROI 화면에서 잘 나오는지 확인하기 위한 부분. 속도 향상을 위해 테스트 외엔 주석 처리
            cvtColor(roi, roi, COLOR_GRAY2BGR);
            for (vector<Vec4i> lines : selected_lines) {
                for (Vec4i l : lines) {
                    line(roi, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 255, 0), 1, LINE_AA);
                }
            }*/

            ld.FindLanes(selected_lines);   // 양 직선들 중 대표 직선(차선)을 결정함
            lanes = ld.FindPos();   // rpos, lpos를 구함

            /*
            // 화면에 차선과 rpos&lpos를 그리는 부분. 속도 향상을 위해 테스트 외엔 주석 처리
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
        // 화면에 결과를 그리는 부분. 속도 향상을 위해 테스트 외엔 주석 처리
        imshow("roi", roi);
        imshow("frame", frame);*/

        waitKey(1);
    }

    output.close();
    cap.release();
    destroyAllWindows();
}
