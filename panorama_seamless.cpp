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
cv::Mat smallRoi;
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

    ptOrigin = cv::Point(x, y); // 初始位置就是ptOrigin
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
      /*matChange2 = matInit(autoSelect);
      cv::imwrite("D:/project/opencv/vs/test/image/roi2.jpg", matChange2);*/
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
  Mat img = imread(IMAGE_PATH_PREFIX + "1.jpg");
  imgs.push_back(img);
  img = imread(IMAGE_PATH_PREFIX + "2.jpg");
  imgs.push_back(img);
  img = imread(IMAGE_PATH_PREFIX + "3.jpg");
  imgs.push_back(img);

  Mat pano; //拼接结果图片
            // Stitcher stitcher = Stitcher::createDefault(try_use_gpu);
  Stitcher stitcher = Stitcher::createDefault(true);
  Stitcher::Status status = stitcher.stitch(imgs, pano);

  if (status != Stitcher::OK) {
    cout << "Can't stitch images, error code = " << int(status) << endl;
    return -1;
  }

  imwrite(result_name, pano);

  while (1) {
    matInit = cv::imread("D:/project/opencv/vs/test/image/result.jpg");
    matSrc = matInit.clone();
    matDst = matInit.clone();
    cv::namedWindow(windowName, CV_WINDOW_NORMAL);

    while (haveselect2 == false) {
      cv::setMouseCallback(windowName, onMouse, NULL);
      cv::imshow(windowName, matSrc);
      cv::waitKey(1);
    }

    //简单的覆盖
    /*matSrc = cv::imread("D:/project/opencv/vs/test/image/result.jpg");
    Mat imageROI = matSrc(rectSelect);
    imwrite("D:/project/opencv/vs/test/image/roi2.jpg", matChange2);
    Mat mask = imread("D:/project/opencv/vs/test/image/roi2.jpg", 0);

    Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
    Mat matChange2_dilate;
    dilate(matChange2, matChange2_dilate, element);

    matChange2_dilate.copyTo(imageROI, mask);
    imwrite("D:/project/opencv/vs/test/image/result.jpg", matSrc);*/
    /*cv::imshow(windowName, matSrc);
    cv::waitKey(2000);*/

    //泊松
    matInit = cv::imread("D:/project/opencv/vs/test/image/result.jpg");
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
    const Point *polygons[1] = {poly[0]}; //常指针指向数组起始点
    int num_points[] = {4};
    fillPoly(src_mask, polygons, num_points, 1, Scalar(255, 255, 255));
    seamlessClone(matChange2, matInit, src_mask, center, output0, NORMAL_CLONE);
    namedWindow("only poisson", CV_WINDOW_NORMAL);
    imshow("only poisson", output0);
    imwrite("D:/project/opencv/vs/test/image/only_poisson.jpg", output0);
    //泊松后色差调整

    Rect rectChange;
    rectChange.x = rectSelect.x - 0.1 * rectSelect.width;
    rectChange.y = rectSelect.y - 0.1 * rectSelect.height;
    rectChange.height = 1.2 * rectSelect.height;
    rectChange.width = 1.2 * rectSelect.width;
    rectChange &= cv::Rect(0, 0, matSrc.cols, matSrc.rows);

    Mat comp = matInit(rectSelect);
    cvtColor(comp, comp, CV_BGR2HSV);

    for (int row = 0; row < comp.rows; row++) {
      for (int col = 0; col < comp.cols; col++) {
        if (comp.at<Vec3b>(row, col).val[2] < 130) {
          comp.at<Vec3b>(row, col).val[2] = 150;
        }
      }
    }
    cvtColor(comp, comp, CV_HSV2BGR);
    Mat element = getStructuringElement(MORPH_RECT, Size(1, 1));
    dilate(comp, comp, element);
    // cv::GaussianBlur(matInit(rectChange), matInit(rectChange), Size(5, 5), 0,
    // 0);
    seamlessClone(matChange2, matInit, src_mask, center, output, NORMAL_CLONE);

    /*Rect rectChange2;
    rectChange2.x = selfSelect.x - 0.05 * selfSelect.width;
    rectChange2.y = selfSelect.y - 0.05 * selfSelect.height;
    rectChange2.height = 1.1 * selfSelect.height;
    rectChange2.width = 1.1 * selfSelect.width;
    rectChange2 &= cv::Rect(0, 0, matSrc.cols, matSrc.rows);*/
    // cv::GaussianBlur(matInit(rectChange2), matInit(rectChange2), Size(15,
    // 15), 0, 0);

    seamlessClone(matChange2, output, src_mask, center, output, NORMAL_CLONE);

    namedWindow("dilate + poisson", CV_WINDOW_NORMAL);
    imshow("dilate + poisson", output);
    imwrite("D:/project/opencv/vs/test/image/dilate_poisson.jpg", output);

    cv::imshow(windowName, output);
    cv::waitKey(2000);
    imwrite("D:/project/opencv/vs/test/image/result.jpg", output);
    haveselect2 = false;
    haveselect = false;
    ptOrigin = cv::Point(-1, -1);
    ptCurrent = cv::Point(-1, -1);
  }
  cv::waitKey(0);

  return 0;
}
