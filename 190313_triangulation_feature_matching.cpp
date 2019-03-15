// 2019.03.14 ȭ������ ��Ī���� ���� +���� �� 2019.03.11
#include "opencv_all.hpp"
#include "opencv2/opencv.hpp" 
#include <cstdio>
#include <iostream>  
//#include "opencv2/opencv.hpp"  
//#include <opencv2/sfm/numeric.hpp>		// ���� �߰��Ѱ�...(���Ƿ�) eigen/core�� ���ٴµ�...

using namespace cv;
using namespace std;

int main(void)
{
	double camera_focal = 1000;
	cv::Point2d camera_center(320, 240);

	// Load two views of 'box.xyz'
	// c.f. You need to run 'image_formation.cpp' to generate point observation.
	//      You can apply Gaussian noise by change value of 'camera_noise' if necessary.
	std::vector<cv::Point2d> points0, points1;
	//	cv::Mat image1 = cv::imread("data/hill01.jpg");	// image �� �� �о����			 ../data/���� or data/���� ���� �޶�...
	//	cv::Mat image2 = cv::imread("data/hill02.jpg");	// ���߿� ���� �� �غ���

	double scale = 0.3;
	printf("load ? \n");
	//cv::resize(image1_o, image1, cv::Size(image1_o.cols*scale, image1_o.rows*scale), 0, 0, 0);  //CV_INTER_NN �̰� ����
	//cv::resize(image2_o, image2, cv::Size(image2_o.cols*scale, image2_o.rows*scale), 0, 0, 0);
	cv::Mat image1 = cv::imread("data/sample05.jpg");	// ���� �� �غ����µ� �߾ȵȴ�... ����ã��
	cv::Mat image2 = cv::imread("data/sample06.jpg");	// 
	if (image1.empty() || image2.empty()) return -1;
	cv::resize(image1, image1, cv::Size(image1.cols*scale, image1.rows*scale), 0, 0, 0);
	cv::resize(image2, image2, cv::Size(image2.cols*scale, image2.rows*scale), 0, 0, 0);
	printf("load complete");

	cv::Ptr<cv::FeatureDetector> fdetector = cv::BRISK::create();	// feature detector(������ �ڳ� ����)
	std::vector<cv::KeyPoint> keypoint1, keypoint2;
	cv::Mat descriptor1, descriptor2;

	fdetector->detectAndCompute(image1, cv::Mat(), keypoint1, descriptor1);  // 2048���� ������ ����...(�̰� �����ϰ� 100�������� ������ �ϴ�)
	fdetector->detectAndCompute(image2, cv::Mat(), keypoint2, descriptor2);
	cv::Ptr<cv::DescriptorMatcher> fmatcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
	std::vector<cv::DMatch> match;
	fmatcher->match(descriptor1, descriptor2, match);

	//	std::vector<cv::Point2f> points1, points2;
	for (size_t i = 0; i < match.size(); i++){
		points0.push_back(keypoint1.at(match.at(i).queryIdx).pt);
		points1.push_back(keypoint2.at(match.at(i).trainIdx).pt);
	}
	cv::Mat inlier_mask;
	//cv::Mat H = cv::findHomography(points1, points0, inlier_mask, cv::RANSAC);

	// ������ �� �� ���� ���� fundamental matrix ���ϱ�(Homography, a_skew �ʿ�)
	//std::vector<cv::Point2d> x_ = points1.operator[];	vector�� at.()��...
	//cv::Mat x_s = cv::sfm::skew(x_);	// �갡 �ȵ���...
	cv::Mat H = cv::findHomography(points0, points1);
	double a1 = points0.at(0).x;
	double a2 = points0.at(0).y;
	double a3 = 1;
	cv::Mat a_skew = (cv::Mat_<double>(3, 3) << 0, -a3, a2, a3, 0, -a1, -a2, a1, 0);
	cv::Mat F_ = a_skew*H;

	// Esitmate relative pose of two views
	cv::Mat F = cv::findFundamentalMat(points0, points1, cv::FM_8POINT);	// homogeneous�� �����ϰ� Ǫ�� �ǵ�?
	cv::Mat K = (cv::Mat_<double>(3, 3) << camera_focal, 0, camera_center.x, 0, camera_focal, camera_center.y, 0, 0, 1);
	cv::Mat E = K.t() * F * K;
	cv::Mat R, t;
	cv::recoverPose(E, points0, points1, K, R, t);

	//	cv::Mat check0 = cv::Mat::eye(3, 4, CV_64F);	// eye�� 3*3 ������Ŀ� 0,0,0 ���߰�
	// Reconstruct 3D points of 'box.xyz' (triangulation)
	cv::Mat P0 = K * cv::Mat::eye(3, 4, CV_64F);	// Mat Ŭ���� ����� �����ϳ�...
	cv::Mat Rt, X;
	//	cout << P0.at<double >(0, 0) << endl;			// ��� �ȵ�... ��������� Ȯ����
	cv::hconcat(R, t, Rt);
	cv::Mat P1 = K * Rt;
	cv::triangulatePoints(P0, P1, points0, points1, X);
	X.row(0) = X.row(0) / X.row(3);
	X.row(1) = X.row(1) / X.row(3);
	X.row(2) = X.row(2) / X.row(3);
	X.row(3) = 1;
	printf("on the process\n");
	// Store the 3D points
	FILE* fout = fopen("lys_triangulation_.xyz", "wt");
	if (fout == NULL) return -1;
	printf("on the process\n");
	for (int c = 0; c < X.cols; c++)
		//printf("working");
		fprintf(fout, "%f %f %f\n", X.at<double>(0, c), X.at<double>(1, c), X.at<double>(2, c));
	fclose(fout);		// â���� ����ϴ°� �ƴ϶� ���Ϸ� ����Ǿ�����(box�� ������ ���� ��ġ)


	// Show the merged image
	cv::Mat original, matched;

	cv::drawMatches(image1, keypoint1, image2, keypoint2, match, matched, cv::Scalar::all(-1), cv::Scalar::all(-1), inlier_mask);
	cv::hconcat(image1, image2, original);		// hconcat : horizontal concat
	cv::vconcat(original, matched, matched);	// vconcat : vertical concat
	//cv::vconcat(matched, merged, merged);
	cv::imshow("3DV Tutorial: Image Stitching", matched);
	cv::waitKey(0);



	// ������� ���� �� �� ���� �غ���
	cv::Mat P0_1T = P0.row(0);
	cv::Mat P0_2T = P0.row(1);
	cv::Mat P0_3T = P0.row(2);
	cv::Mat P1_1T = P1.row(0);
	cv::Mat P1_2T = P1.row(1);
	cv::Mat P1_3T = P1.row(2);
	//cv::Mat A(4,4,CV_64F);

	std::vector<cv::Point3d> X_real;
	for (int i = 0; i <points0.size(); i++)
		//	for (int i = 0; i <1; i++)
	{
		cv::Mat A;
		double x0 = points0[i].x;
		double x1 = points1[i].x;
		double y0 = points0[i].y;
		double y1 = points1[i].y;
		cv::Mat A0 = x0*P0_3T - P0_1T;
		cv::Mat A1 = y0*P0_3T - P0_2T;
		cv::Mat A2 = x1*P0_3T - P1_1T;
		cv::Mat A3 = y1*P0_3T - P1_2T;
		std::vector<cv::Point3f> X_temp;
		A.push_back(A0);
		A.push_back(A1);
		A.push_back(A2);
		A.push_back(A3);		// ��������� Ȯ���ϴµ�, A�� �� ������°���...
		// A�� singular value ���ϴ°� �ʿ�.
		cv::Mat matrW(4, 1, CV_64F);
		cv::Mat matrU(4, 4, CV_64F);
		cv::Mat matrV(4, 4, CV_64F);	// matrV�� �´� ���� �����°� Ȯ�ε�
		cv::SVD::compute(A, matrW, matrU, matrV);
		// singular value�� ������ X_ �� �ʿ�
		// ���� : cvmSet(mat1, 0, 0, 2.0f); // mat1 ��Ʈ������ (0, 0) �׿� 2.0f ���� ����ִ´�
		printf("once\n");
		// Mat�� ��� ������ �߸��ȵ�...

		printf("%f\n", matrV.at<double>(0, 0));
		// vector���ٰ� �ϳ��� ���� �߰��� �����ϳİ�....
		X_real.push_back(cv::Point3d(matrV.at<double>(3, 0) / matrV.at<double>(3, 3), matrV.at<double>(3, 1) / matrV.at<double>(3, 3), matrV.at<double>(3, 2) / matrV.at<double>(3, 3)));
		//	X_real[i].x = 3.0;												// ��ǥ�δ� ���ö��� ����. �� ����ÿ��� push_back
		//	X_real[i].y = matrV.at<float>(3, 1) / matrV.at<float>(3, 3);
		//	X_real[i].z = matrV.at<float>(3, 2) / matrV.at<float>(3, 3);
	}

	FILE* fout1 = fopen("lys_triangulation__.xyz", "wt");
	if (fout1 == NULL) return -1;
	for (int c = 0; c < points0.size(); c++)
		//printf("working");
		fprintf(fout1, "%f %f %f\n", X_real[c].x, X_real[c].y, X_real[c].z);
	fclose(fout1);		// â���� ����ϴ°� �ƴ϶� ���Ϸ� ����Ǿ�����(box�� ������ ���� ��ġ)
	printf("done\n");

	return 0;
}
