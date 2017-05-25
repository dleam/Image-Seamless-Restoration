#include <iostream>
#include <opencv2/opencv.hpp>
#include <stitching.hpp>

using namespace std;
using namespace cv;

bool haveselect = false;
bool haveselect2 = false;
string IMAGE_PATH_PREFIX = "D:/project/opencv/vs/test/image/";
string result_name = IMAGE_PATH_PREFIX + "result.jpg";
vector<Mat> imgs;
cv::Mat matInit;
cv::Mat matSrc;
cv::Mat matDst;
cv::Mat matChange, matChange2;
cv::Rect rectChange;
cv::Rect rectSelect;
cv::Rect autoSelect;
cv::Rect selfSelect;
cv::Point ptOrigin = cv::Point(-1, -1);
cv::Point ptCurrent = cv::Point(-1, -1);
std::string windowName = "Panorama";

void onMouse(int event, int x, int y, int flags, void *param) {
  if (matSrc.data == NULL)
    return;

  if (x > matSrc.cols || y > matSrc.rows)
    return;

  int thickness = 5;
  int lineType = CV_AA;

  if (event == CV_EVENT_LBUTTONDOWN) {

    ptOrigin = cv::Point(x, y);
    matInit.copyTo(matSrc);
    matInit.copyTo(matDst);
    if (haveselect == false) {
      rectSelect = cv::Rect(x, y, 0, 0);

      cv::circle(matSrc, ptOrigin, 1, cv::Scalar(0, 0, 255), CV_FILLED,
                 lineType);
      cv::imshow(windowName, matSrc);
      matSrc.copyTo(matDst);

    } else {

      cv::Point ptfixs;
      ptfixs.x = rectSelect.x;
      ptfixs.y = rectSelect.y;
      cv::Point ptfixe;
      ptfixe.x = rectSelect.x + rectSelect.width;
      ptfixe.y = rectSelect.y + rectSelect.height;
      cv::rectangle(matSrc, ptfixs, ptfixe, cv::Scalar(0, 0, 0xFF), thickness,
                    lineType);

      selfSelect.x = ptCurrent.x;
      selfSelect.y = ptCurrent.y;
      selfSelect.height = rectSelect.height;
      selfSelect.width = rectSelect.width;
      selfSelect &= cv::Rect(0, 0, matSrc.cols, matSrc.rows);

      cv::Point ptAuto;
      ptAuto.x = selfSelect.x + selfSelect.width;
      ptAuto.y = selfSelect.y + selfSelect.height;
      cv::rectangle(matSrc, ptCurrent, ptAuto, cv::Scalar(0, 0xFF, 0),
                    thickness, lineType);
      cv::imshow(windowName, matSrc);
    }

  } else if (event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON)) {

    matDst.copyTo(matSrc);
    ptCurrent = cv::Point(x, y);

    if (haveselect == false) {
      rectSelect.x = MIN(ptOrigin.x, ptCurrent.x);
      rectSelect.y = MIN(ptOrigin.y, ptCurrent.y);
      rectSelect.height = abs(ptOrigin.y - ptCurrent.y);
      rectSelect.width = abs(ptOrigin.x - ptCurrent.x);
      rectSelect &= cv::Rect(0, 0, matSrc.cols, matSrc.rows);

      cv::rectangle(matSrc, ptOrigin, ptCurrent, cv::Scalar(0, 0xFF, 0),
                    thickness, lineType);
      cv::imshow(windowName, matSrc);
    }

    else {

      cv::Point ptfixs;
      ptfixs.x = rectSelect.x;
      ptfixs.y = rectSelect.y;
      cv::Point ptfixe;
      ptfixe.x = rectSelect.x + rectSelect.width;
      ptfixe.y = rectSelect.y + rectSelect.height;
      cv::rectangle(matSrc, ptfixs, ptfixe, cv::Scalar(0, 0, 0xFF), thickness,
                    lineType);

      selfSelect.x = ptCurrent.x;
      selfSelect.y = ptCurrent.y;
      selfSelect.height = rectSelect.height;
      selfSelect.width = rectSelect.width;
      selfSelect &= cv::Rect(0, 0, matSrc.cols, matSrc.rows);

      cv::Point ptAuto;
      ptAuto.x = selfSelect.x + selfSelect.width;
      ptAuto.y = selfSelect.y + selfSelect.height;
      cv::rectangle(matSrc, ptCurrent, ptAuto, cv::Scalar(0, 0xFF, 0),
                    thickness, lineType);
      cv::imshow(windowName, matSrc);
    }

  } else if (event == CV_EVENT_LBUTTONUP) {

    ptCurrent = cv::Point(x, y);

    if (haveselect == false) {
      rectSelect.x = MIN(ptOrigin.x, ptCurrent.x);
      rectSelect.y = MIN(ptOrigin.y, ptCurrent.y);
      rectSelect.height = abs(ptOrigin.y - ptCurrent.y);
      rectSelect.width = abs(ptOrigin.x - ptCurrent.x);
      rectSelect &= cv::Rect(0, 0, matSrc.cols, matSrc.rows);

      cv::rectangle(matSrc, ptOrigin, ptCurrent, cv::Scalar(0, 0, 0xFF),
                    thickness, lineType);
      cv::imshow(windowName, matSrc);

      matChange = matInit(rectSelect);
      cv::imwrite("D:/project/opencv/vs/test/image/roi.jpg", matChange);

      autoSelect.x = rectSelect.x - rectSelect.width;
      autoSelect.y = rectSelect.y;
      autoSelect.width = rectSelect.width;
      autoSelect.height = rectSelect.height;
      autoSelect &= cv::Rect(0, 0, matSrc.cols, matSrc.rows);
      haveselect = true;

    }

    else {

      cv::Point ptfixs;
      ptfixs.x = rectSelect.x;
      ptfixs.y = rectSelect.y;
      cv::Point ptfixe;
      ptfixe.x = rectSelect.x + rectSelect.width;
      ptfixe.y = rectSelect.y + rectSelect.height;
      cv::rectangle(matSrc, ptfixs, ptfixe, cv::Scalar(0, 0, 0xFF), thickness,
                    lineType);

      selfSelect.x = ptCurrent.x;
      selfSelect.y = ptCurrent.y;
      selfSelect.height = rectSelect.height;
      selfSelect.width = rectSelect.width;
      selfSelect &= cv::Rect(0, 0, matSrc.cols, matSrc.rows);
      cv::Point ptAuto;
      ptAuto.x = selfSelect.x + selfSelect.width;
      ptAuto.y = selfSelect.y + selfSelect.height;
      cv::rectangle(matSrc, ptCurrent, ptAuto, cv::Scalar(0, 0xFF, 0),
                    thickness, lineType);
      cv::imshow(windowName, matSrc);
      matChange2 = matInit(selfSelect);
      cv::imwrite("D:/project/opencv/vs/test/image/roi2.jpg", matChange2);
      haveselect2 = true;
    }
  }
}

static void getBinMask(const Mat &comMask, Mat &binMask) {
  binMask.create(comMask.size(), CV_8UC1);
  binMask = comMask & 1;
}

int main() {
  // use your own pictures here to make the panorama if yout want to create it
  // here.
  // Mat img = imread(IMAGE_PATH_PREFIX + "1.jpg");
  // imgs.push_back(img);
  // img = imread(IMAGE_PATH_PREFIX + "2.jpg");
  // imgs.push_back(img);
  // img = imread(IMAGE_PATH_PREFIX + "3.jpg");
  // imgs.push_back(img);
  // Mat pano; // Stitcher stitcher = Stitcher::createDefault(try_use_gpu);
  // Stitcher stitcher = Stitcher::createDefault(true);
  // Stitcher::Status status = stitcher.stitch(imgs, pano);
  // if (status != Stitcher::OK) {
  //   cout << "Can't stitch images, error code = " << int(status) << endl;
  //   return -1;
  // }
  // imwrite(result_name, pano);

  while (1) {
    // change your path to the panorama image
    string path_to_panorama = "D:/project/opencv/vs/test/image/result.jpg";
    matInit = cv::imread(path_to_panorama);
    matSrc = matInit.clone();
    matDst = matInit.clone();
    cv::namedWindow(windowName, CV_WINDOW_NORMAL);

    while (haveselect2 == false) {
      cv::setMouseCallback(windowName, onMouse, NULL);
      cv::imshow(windowName, matSrc);
      cv::waitKey(1);
    }

    // Poisson
    matInit = cv::imread(path_to_panorama);
    cv::Mat output0, output;
    Point center;
    center.x = rectSelect.x + rectSelect.width / 2;
    center.y = rectSelect.y + rectSelect.height / 2;
    Mat src_mask =
        Mat::zeros(matChange2.rows, matChange2.cols, matChange2.depth());
    Point poly[1][4];
    poly[0][0] = Point(0, 0);
    poly[0][1] = Point(0 + rectSelect.width, 0);
    poly[0][2] = Point(0 + rectSelect.width, 0 + rectSelect.height);
    poly[0][3] = Point(0, 0 + rectSelect.height);
    const Point *polygons[1] = {poly[0]};
    int num_points[] = {4};
    fillPoly(src_mask, polygons, num_points, 1, Scalar(255, 255, 255));
    seamlessClone(matChange2, matInit, src_mask, center, output0, NORMAL_CLONE);
    imwrite(path_to_panorama, output0);
    cv::waitKey(2000);

    haveselect2 = false;
    haveselect = false;
    ptOrigin = cv::Point(-1, -1);
    ptCurrent = cv::Point(-1, -1);
  }
  cv::waitKey(0);

  return 0;
}
