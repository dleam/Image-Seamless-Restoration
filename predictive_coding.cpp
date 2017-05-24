#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <cv.hpp>



cv::Mat matSrc;
cv::Rect    rectSelect;
cv::Mat matInit;
cv::Point   ptOrigin = cv::Point(-1, -1);
cv::Point   ptCurrent = cv::Point(-1, -1);

void onMouse(int event, int x, int y, int flags, void *param)
{
	if (matSrc.data == NULL)
		return;

	if (x > matSrc.cols || y > matSrc.rows)
		return;


	int     thickness = 5;
	int     lineType = CV_AA;

	if (event == CV_EVENT_LBUTTONDOWN) {

		ptOrigin = cv::Point(x, y);// 初始位置就是ptOrigin
		matInit.copyTo(matSrc);

		rectSelect = cv::Rect(x, y, 0, 0);
		cv::circle(matSrc, ptOrigin, 1, cv::Scalar(0, 0, 255), CV_FILLED, lineType);
		cv::imshow(windowName, matSrc);






	}
	else if (event == CV_EVENT_MOUSEMOVE &&
		(flags & CV_EVENT_FLAG_LBUTTON)) {

			ptCurrent = cv::Point(x, y);
			rectSelect.x = MIN(ptOrigin.x, ptCurrent.x);
			rectSelect.y = MIN(ptOrigin.y, ptCurrent.y);
			rectSelect.height = abs(ptOrigin.y - ptCurrent.y);
			rectSelect.width = abs(ptOrigin.x - ptCurrent.x);
			rectSelect &= cv::Rect(0, 0, matSrc.cols, matSrc.rows);
			cv::rectangle(matSrc, ptOrigin, ptCurrent, cv::Scalar(0, 0xFF, 0),thickness, lineType);
			cv::imshow(windowName, matSrc);




	}
	else if (event == CV_EVENT_LBUTTONUP) {

			ptCurrent = cv::Point(x, y);
			rectSelect.x = MIN(ptOrigin.x, ptCurrent.x);
			rectSelect.y = MIN(ptOrigin.y, ptCurrent.y);
			rectSelect.height = abs(ptOrigin.y - ptCurrent.y);
			rectSelect.width = abs(ptOrigin.x - ptCurrent.x);
			rectSelect &= cv::Rect(0, 0, matSrc.cols, matSrc.rows);
			cv::rectangle(matSrc, ptOrigin, ptCurrent, cv::Scalar(0, 0, 0xFF),thickness, lineType);
			cv::imshow(windowName, matSrc);
			matChange = matInit(rectSelect);

















	}

}

struct VF          //value and frequency
{
	int value;
	int frequency;
};
void getHist(cv::Mat&img, int* hist)
{
	CV_Assert(img.depth() == CV_8U);
	for (int r = 0; r != img.rows; ++r)
	{
		const uchar *p = img.ptr<uchar>(r);
		for (int c = 0; c != img.cols; ++c)
		{
			hist[p[c]] += 1;
		}
	}
}
void getHist_prediction(cv::Mat_<int> & img, cv::Mat &img_source, std::vector<int> &img_vector, std::vector<VF> &v)//, std::vector<std::vector<int>> &hist,std::vector<int> &img_vector )
{
	for (int i = 0; i != img_source.rows; ++i)
	{
		for (int j = 0; j != img_source.cols; ++j)
		{
			img_vector.push_back(img.at<int>(i, j));
		}
	}
	VF va;//temp
	VF vr;
	va.value = img_vector[0];//initial
	va.frequency = 0;
	v.push_back(va);

	for (int i = 0; i != img_vector.size(); i++)
	{
		int k;
		for (k = 0; k != v.size(); ++k)
		{
			if (img_vector[i] == v[k].value)
			{
				break;
			}
		}
		if (k != v.size())
		{
			v[k].frequency += 1;
		}
		else
		{
			vr.value = img_vector[i];
			vr.frequency = 1;
			v.push_back(vr);
		}
	}

}
double entropy_calculation(cv::Mat&img, int* hist)
{
	float ent = 0, N = static_cast<float>(img.rows*img.cols);
	for (int i = 0; i != 256; ++i)
	{
		if (hist[i] != 0) {
			float p = hist[i] / N;
			ent += p*log(p) / log(2);
		}
	}
	ent = -ent;
	return (ent);
}
double entropy_prediction(cv::Mat &img, std::vector<VF> &v)
{
	float ent = 0, N = static_cast<float>(img.rows*img.cols);
	for (int i = 0; i != v.size(); ++i)
	{
		float p = v[i].frequency / N;
		ent += p*log(p) / log(2);
	}
	ent = -ent;
	return (ent);
}
void vertical_predict(cv::Mat &img, cv::Mat &result, cv::Mat_<int> &prediction)
{
	//give the first col initial value
	for (int i = 0; i != img.rows; ++i)
	{
		{
			result.at<uchar>(i, 0) = img.at<uchar>(i, 0);
			prediction.at<int>(i, 0) = img.at<uchar>(i, 0);
		}
	}
	//predict the rows value step by step
	for (int i = 1; i != img.rows; ++i)
	{
		for (int j = 0; j != img.cols; ++j)
		{
			result.at<uchar>(i, j) = img.at<uchar>(i - 1, j);
			prediction.at<int>(i, j) = img.at<uchar>(i - 1, j);
		}
	}
	//e=x-y
	for (int i = 0; i != img.rows; ++i)
	{
		for (int j = 0; j != img.cols; ++j)
		{
			prediction.at<int>(i, j) = img.at<uchar>(i, j) - result.at<uchar>(i, j);
			result.at<uchar>(i, j) = int((img.at<uchar>(i, j) - result.at<uchar>(i, j) + 255) / 2);//normalization (-225,255)->(0,255)

		}
	}
}
void horizontal_predict(cv::Mat &img, cv::Mat &result, cv::Mat_<int> &prediction)
{
	for (int j = 0; j != img.cols; ++j)
	{
		{
			result.at<uchar>(0, j) = img.at<uchar>(0, j);
			prediction.at<int>(0, j) = img.at<uchar>(0, j);
		}
	}
	for (int j = 1; j != img.cols; ++j)
	{
		for (int i = 0; i != img.rows; ++i)
		{
			result.at<uchar>(i, j) = img.at<uchar>(i, j - 1);
			prediction.at<int>(i, j) = img.at<uchar>(i, j - 1);
		}
	}
	//e=x-y
	for (int i = 0; i != img.rows; ++i)
	{
		for (int j = 0; j != img.cols; ++j)
		{
			prediction.at<int>(i, j) = img.at<uchar>(i, j) - result.at<uchar>(i, j);
			result.at<uchar>(i, j) = int((img.at<uchar>(i, j) - result.at<uchar>(i, j) + 255) / 2);//normalization (-225,255)->(0,255)

		}
	}
}
void adaptive_predict(cv::Mat &img, cv::Mat &result, cv::Mat_<int> &prediction, cv::Mat_<int> &direction)
{
	for (int i = 0; i != img.rows; ++i)//initial first row
	{
		{
			result.at<uchar>(i, 0) = int(img.at<uchar>(i, 0));
			prediction.at<int>(i, 0) = img.at<uchar>(i, 0);
		}
	}
	for (int j = 0; j != img.cols; ++j)//initial first col
	{
		{
			result.at<uchar>(0, j) = int(img.at<uchar>(0, j));
			prediction.at<int>(0, j) = img.at<uchar>(0, j);
		}
	}
	for (int j = 1; j != img.cols; ++j)
	{
		for (int i = 1; i != img.rows; ++i)
		{                        //compare (i,j) itself with the neighbor
			if (abs(img.at<uchar>(i, j) - img.at<uchar>(i, j - 1)) < abs(img.at<uchar>(i, j) - img.at<uchar>(i - 1, j)))
			{
				result.at<uchar>(i, j) = img.at<uchar>(i, j - 1);
				prediction.at<int>(i, j) = img.at<uchar>(i, j - 1);
				direction.at<int>(i - 1, j - 1) = 0;//0 represent row prediction(size is smaller than img by 1 row and col)
			}
			else
			{
				result.at<uchar>(i, j) = img.at<uchar>(i - 1, j);
				prediction.at<int>(i, j) = img.at<uchar>(i - 1, j);
				direction.at<int>(i - 1, j - 1) = 1;//1 represent row prediction
			}
		}
	}
	//e=x-y
	for (int i = 0; i != img.rows; ++i)
	{
		for (int j = 0; j != img.cols; ++j)
		{
			prediction.at<int>(i, j) = img.at<uchar>(i, j) - result.at<uchar>(i, j);
			result.at<uchar>(i, j) = int((img.at<uchar>(i, j) - result.at<uchar>(i, j) + 255) / 2);//normalization (-225,255)->(0,255)

		}
	}
}
void rule_predict(cv::Mat &img, cv::Mat &result, cv::Mat_<int> &prediction)
{
	for (int i = 0; i != img.rows; ++i)//initial first row
	{
		{
			result.at<uchar>(i, 0) = int(img.at<uchar>(i, 0));
			prediction.at<int>(i, 0) = img.at<uchar>(i, 0);
		}
	}
	for (int j = 0; j != img.cols; ++j)//initial first col
	{
		{
			result.at<uchar>(0, j) = int(img.at<uchar>(0, j));
			prediction.at<int>(0, j) = img.at<uchar>(0, j);
		}
	}
	for (int j = 1; j != img.cols; ++j)
	{
		for (int i = 1; i != img.rows; ++i)
		{
			if (abs(img.at<uchar>(i - 1, j) - img.at<uchar>(i - 1, j - 1)) < abs(img.at<uchar>(i, j - 1) - img.at<uchar>(i - 1, j - 1)))
			{
				result.at<uchar>(i, j) = img.at<uchar>(i, j - 1);
				prediction.at<int>(i, j) = img.at<uchar>(i, j - 1);
			}
			else
			{
				result.at<uchar>(i, j) = img.at<uchar>(i - 1, j);
				prediction.at<int>(i, j) = img.at<uchar>(i - 1, j);
			}
		}
	}
	//e=x-y
	for (int i = 0; i != img.rows; ++i)
	{
		for (int j = 0; j != img.cols; ++j)
		{
			prediction.at<int>(i, j) = img.at<uchar>(i, j) - result.at<uchar>(i, j);
			result.at<uchar>(i, j) = int((img.at<uchar>(i, j) - result.at<uchar>(i, j) + 255) / 2);//normalization (-225,255)->(0,255)

		}
	}
}


int main()
{
	std::vector<VF> v_h, v_v, v_a, v_r, v_d;
	std::vector<int> img_vector;
	double entropy_value = 0;
	int hist_s[256] = { 0 };
	cv::Mat_<int> direction;//the direction Mat in adaptive prediction
	cv::Mat img, horizontal_img, vertical_img, adaptive_img, rule_img;
	cv::Mat_<int> horizontal_prediction, vertical_prediction, adaptive_prediction, rule_prediction;
	img = cv::imread("D:/USTC/Image analyze/ppt/image/source.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	horizontal_img = cv::Mat::zeros(img.size(), CV_8U);
	vertical_img = cv::Mat::zeros(img.size(), CV_8U);
	adaptive_img = cv::Mat::zeros(img.size(), CV_8U);
	rule_img = cv::Mat::zeros(img.size(), CV_8U);
	direction = cv::Mat::zeros(img.rows - 1, img.cols - 1, CV_8S);
	horizontal_prediction = cv::Mat::zeros(img.size(), CV_8S);
	vertical_prediction = cv::Mat::zeros(img.size(), CV_8S);
	adaptive_prediction = cv::Mat::zeros(img.size(), CV_8S);
	rule_prediction = cv::Mat::zeros(img.size(), CV_8S);
	/*show the image of normalization*/
	horizontal_predict(img, horizontal_img, horizontal_prediction);
	vertical_predict(img, vertical_img, vertical_prediction);
	adaptive_predict(img, adaptive_img, adaptive_prediction, direction);
	rule_predict(img, rule_img, rule_prediction);
	imshow("horizontal_img", horizontal_img);
	imshow("vertical_img", vertical_img);
	imshow("adaptive_img", adaptive_img);
	imshow("rule_img", rule_img);
	cv::imwrite("D:\\USTC\\Image analyze\\ppt\\code\\horizontal_img.jpg", horizontal_img);
	cv::imwrite("D:\\USTC\\Image analyze\\ppt\\code\\vertical_img.jpg", vertical_img);
	cv::imwrite("D:\\USTC\\Image analyze\\ppt\\code\\adaptive_img.jpg", adaptive_img);
	cv::imwrite("D:\\USTC\\Image analyze\\ppt\\code\\rule_img.jpg", rule_img);
	/*calculate the entropy of the source image and different kinds of predictive image*/

	getHist(img, hist_s);//normal Hist calculation
	entropy_value = entropy_calculation(img, hist_s);
	std::cout << "Entropy of the source image = " << entropy_value << "\n" << std::endl;

	std::cout << "Entropy of the residue value :" << std::endl;
	getHist_prediction(horizontal_prediction, img, img_vector, v_h);//prediction Hist calculation
	entropy_value = entropy_prediction(horizontal_img, v_h);
	std::cout << "horizontal prediction = " << entropy_value << std::endl;
	img_vector.clear();
	getHist_prediction(vertical_prediction, img, img_vector, v_v);
	entropy_value = entropy_prediction(vertical_img, v_v);
	std::cout << "vertical prediction = " << entropy_value << std::endl;
	img_vector.clear();

	getHist_prediction(adaptive_prediction, img, img_vector, v_a);
	entropy_value = entropy_prediction(adaptive_img, v_a);
	std::cout << "adaptive prediction with overhead = " << entropy_value;
	img_vector.clear();
	getHist_prediction(direction, direction, img_vector, v_d);//direction Hist
	entropy_value = entropy_prediction(direction, v_d);
	std::cout << "  (overhead entropy: " << entropy_value << ")" << std::endl;
	img_vector.clear();




	std::cout << "adaptive prediction without overhead" << std::endl;
	getHist_prediction(rule_prediction, img, img_vector, v_r);
	entropy_value = entropy_prediction(rule_img, v_r);
	std::cout << "First time entropy = " << entropy_value << std::endl;
	img_vector.clear();



	/////twice adaptive prediction without overhead/////////
	std::vector<VF> v_2;
	cv::Mat_<int> rule_prediction_2;
	rule_prediction_2 = cv::Mat::zeros(img.size(), CV_8S);
	cv::Mat rule_img_2;
	rule_img_2 = cv::Mat::zeros(img.size(), CV_8U);
	rule_predict(rule_img, rule_img_2, rule_prediction_2);
	getHist_prediction(rule_prediction_2, img, img_vector, v_2);
	entropy_value = entropy_prediction(rule_img_2, v_2);
	std::cout << "Second time entropy = " << entropy_value << std::endl;
	imshow("rule_img_2", rule_img_2);
	cv::imwrite("D:\\USTC\\Image analyze\\ppt\\code\\rule_img_2.jpg", rule_img_2);
	img_vector.clear();

	/////three times adaptive prediction without overhead/////////
	std::vector<VF> v_3;
	cv::Mat_<int> rule_prediction_3;
	rule_prediction_3 = cv::Mat::zeros(img.size(), CV_8S);
	cv::Mat rule_img_3;
	rule_img_3 = cv::Mat::zeros(img.size(), CV_8U);
	rule_predict(rule_img_2, rule_img_3, rule_prediction_3);
	getHist_prediction(rule_prediction_3, img, img_vector, v_3);
	entropy_value = entropy_prediction(rule_img_3, v_3);
	std::cout << "Third time entropy = " << entropy_value << std::endl;
	imshow("rule_img_3", rule_img_3);
	cv::imwrite("D:\\USTC\\Image analyze\\ppt\\code\\rule_img_3.jpg", rule_img_3);

	img_vector.clear();

	/////four times adaptive prediction without overhead/////////
	std::vector<VF> v_4;
	cv::Mat_<int> rule_prediction_4;
	rule_prediction_4 = cv::Mat::zeros(img.size(), CV_8S);
	cv::Mat rule_img_4;
	rule_img_4 = cv::Mat::zeros(img.size(), CV_8U);
	rule_predict(rule_img_3, rule_img_4, rule_prediction_4);
	getHist_prediction(rule_prediction_4, img, img_vector, v_4);
	entropy_value = entropy_prediction(rule_img_4, v_4);
	std::cout << "Forth time entropy = " << entropy_value << std::endl;
	imshow("rule_img_4", rule_img_4);
	cv::imwrite("D:\\USTC\\Image analyze\\ppt\code\\rule_img_4.jpg", rule_img_4);

	cv::waitKey(0);
}
