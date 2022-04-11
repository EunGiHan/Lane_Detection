#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

void on_level_change(int pos, void* userdata);

int main()
{
	VideoCapture cap("subProject.avi");

	if (!cap.isOpened()) {
		cerr << "Video open failed!" << endl;
		return -1;
	}

	int h = cvRound(cap.get(CAP_PROP_FRAME_HEIGHT));
	int w = cvRound(cap.get(CAP_PROP_FRAME_WIDTH));
    int bottom_y = 460; // 값 확인하기

    int warp_h = h / 2; // 나중에 수정하기
    int warp_w = w / 2; // 나중에 수정하기
    Point warp_dst[4]{
        Point(0, 0),
        Point(0, warp_h),
        Point(warp_w, 0),
        Point(warp_w, warp_h)
    };

    namedWindow("bird_eye_view");
    // x는 왼쪽 기준으로 대칭해서 잡기, y 아래쪽은 문제에 제공됨
    createTrackbar("left_top_x", "bird_eye_view", 0, 640);
    createTrackbar("top_y", "bird_eye_view", 0, 480);
    createTrackbar("left_bottom_x", "bird_eye_view", 0, 640);

    Mat frame, color_warp_area, perspec_mat, warp_img;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // trackbar 받아오기
        Point left_top = Point(getTrackbarPos("left_top_x", "bird_eye_view"), getTrackbarPos("top_y", "bird_eye_view"));
        Point right_top = Point(w - getTrackbarPos("left_top_x", "bird_eye_view"), getTrackbarPos("top_y", "bird_eye_view"));
        Point left_bottom = Point(getTrackbarPos("left_bottom_x", "bird_eye_view"), bottom_y);
        Point right_bottom = Point(w - getTrackbarPos("left_bottom_x", "bird_eye_view"), bottom_y);
        Point warp_src[4]{
            left_top, left_bottom, right_bottom, right_top
        };

        fillPoly(color_warp_area, warp_src, Scalar(0, 255, 0));
        frame = addWeighted(frame, 1, color_warp_area, 0.3, 0);


        perspec_mat = getPerspectiveTransform(warp_src, warp_dst);
        warpPerspective(frame, warp_img, perspec_mat, Size(warp_w, warp_h));

        imshow("frame", frame);
        imshow("bird_eye_view", warp_img);
        if (waitKey(10) == 27)
            break;
    }

    cap.release();
    destroyAllWindows();
}

void on_level_change(int pos, void* userdata) {
    return;
}