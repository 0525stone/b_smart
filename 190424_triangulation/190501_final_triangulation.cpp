#include "opencv_all.hpp"

int main(void)
{
	double camera_focal = 820;
	cv::Point2d camera_center(314, 262);

	// Load two views of 'box.xyz'
	// c.f. You need to run 'image_formation.cpp' to generate point observation.
	//      You can apply Gaussian noise by change value of 'camera_noise' if necessary.
	std::vector<cv::Point2d> points0, points1;
	FILE* fin0 = fopen(" _image_formation0.xyz", "rt");  
	FILE* fin1 = fopen(" _image_formation1.xyz", "rt");
//	FILE* fin0 = fopen("_image000.xyz", "rt");  // _image_formation0.xyz 임 원랜
//	FILE* fin1 = fopen("_image001.xyz", "rt");
	if (fin0 == NULL || fin1 == NULL) return -1;
	while (!feof(fin0) || !feof(fin1))
	{
		double x, y, w;
		if (!feof(fin0) && fscanf(fin0, "%lf %lf %lf", &x, &y, &w) == 3)
			points0.push_back(cv::Point2d(x, y));
		if (!feof(fin1) && fscanf(fin1, "%lf %lf %lf", &x, &y, &w) == 3)
			points1.push_back(cv::Point2d(x, y));
	}
	fclose(fin0);
	fclose(fin1);
	if (points0.size() != points1.size()) return -1;

	// Esitmate relative pose of two views
	cv::Mat F = cv::findFundamentalMat(points0, points1, cv::FM_8POINT);
	cv::Mat K = (cv::Mat_<double>(3, 3) << camera_focal, 0, camera_center.x, 0, camera_focal, camera_center.y, 0, 0, 1);
	cv::Mat E = K.t() * F * K;
	cv::Mat R, t;
	cv::recoverPose(E, points0, points1, K, R, t);
	R = (cv::Mat_<double>(3, 3) << 1, 0,0, 0, 1, 0, 0, 0, 1);
	t = (cv::Mat_<double>(3, 1) << -0.1,0 ,0); // 약 5cm 차이
	// Reconstruct 3D points of 'box.xyz' (triangulation)
	cv::Mat P0 = K * cv::Mat::eye(3, 4, CV_64F);
	cv::Mat Rt, X;
	cv::hconcat(R, t, Rt);
	cv::Mat P1 = K * Rt;
	cv::triangulatePoints(P0, P1, points0, points1, X);
	X.row(0) = X.row(0) / X.row(3);
	X.row(1) = X.row(1) / X.row(3);
	X.row(2) = X.row(2) / X.row(3);
	X.row(3) = 1;

	// Store the 3D points
	FILE* fout = fopen("test/triangulation.xyz", "wt");
	if (fout == NULL) return -1;
	for (int c = 0; c < X.cols; c++)
		fprintf(fout, "%f %f %f\n", X.at<double>(0, c), X.at<double>(1, c), X.at<double>(2, c));
	fclose(fout);
	return 0;
}