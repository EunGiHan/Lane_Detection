#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main()
{
	VideoCapture cap("subProject.avi");

	if (!cap.isOpened()) {
		cerr << "Video open failed!" << endl;
		return -1;
	}

	int h = cvRound(cap.get(CAP_PROP_FRAME_HEIGHT));
	int w = cvRound(cap.get(CAP_PROP_FRAME_WIDTH));

    Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        imshow("frame", frame);
        if (waitKey(10) == 27)
            break;
    }

    cap.release();
    destroyAllWindows();
}
