#include "opencv_all.hpp"

#include <iostream>  
#include "opencv2/opencv.hpp"  
#include <typeinfo>

using namespace cv;
using namespace std;

int main(void)
{
	// VideoCapture�� Video������ �ҷ���
	VideoCapture vc(0); if (!vc.isOpened())return 0; // connection failed

	// �ʿ亯�� ���� Ķ���극�̼� ���
	cv::Mat K = (cv::Mat_<double>(3, 3) << 640.3783, 0, 316.6393, 0, 679.3930, 227.904493, 0, 0, 1);
	//	cv::Mat dist_coeff = (cv::Mat_<double>(4, 1) << 0.004628, -0.044073, 0.008047, 0.001567);
	cv::Mat dist_coeff = (cv::Mat_<double>(4, 1) << 0.056178, -0.213159, -0.029619, 0.014168);
	Mat img, imgr, imgt;  // ���� �޾��� Matrix ����
	Size boardSize(4, 7); // chessboard ������ (üũ���� ���� ����)
	float board_cellsize = 26.0;

	// �������� ť�� ���� ���� (projection ���� Ȯ�ο�)
	std::vector<cv::Point3d> box_lower, box_upper;
	box_lower.push_back(cv::Point3d(3 * board_cellsize, 2 * board_cellsize, 0));
	box_lower.push_back(cv::Point3d(5 * board_cellsize, 2 * board_cellsize, 0));
	box_lower.push_back(cv::Point3d(5 * board_cellsize, 4 * board_cellsize, 0));
	box_lower.push_back(cv::Point3d(3 * board_cellsize, 4 * board_cellsize, 0));
	box_upper.push_back(cv::Point3d(3 * board_cellsize, 2 * board_cellsize, -2 * board_cellsize));
	box_upper.push_back(cv::Point3d(5 * board_cellsize, 2 * board_cellsize, -2 * board_cellsize));
	box_upper.push_back(cv::Point3d(5 * board_cellsize, 4 * board_cellsize, -2 * board_cellsize));
	box_upper.push_back(cv::Point3d(3 * board_cellsize, 4 * board_cellsize, -2 * board_cellsize));

	//��ķ���� ĸ�ĵǴ� �̹��� ũ�⸦ ������
	Size size = Size((int)vc.get(CAP_PROP_FRAME_WIDTH),
		(int)vc.get(CAP_PROP_FRAME_HEIGHT));

	//���Ϸ� �������� �����ϱ� ���� �غ�  
	VideoWriter outputVideo;
	outputVideo.open("ouput.avi", VideoWriter::fourcc('X', 'V', 'I', 'D'),
		15, size, true);
	if (!outputVideo.isOpened())
	{
		cout << "�������� �����ϱ� ���� �ʱ�ȭ �۾� �� ���� �߻�" << endl;
		return 1;
	}

	cv::Mat ref_img, X;
	// �̹��� ó�� / ���� ����

	// ���� for camera pose
	std::vector<cv::Point3f>  obj_points;
	for (int r = 0; r < boardSize.height; r++)
	for (int c = 0; c < boardSize.width; c++)
		obj_points.push_back(cv::Point3d(board_cellsize * c, board_cellsize * r, 0));

	cv::Mat ref_rvec, ref_tvec, ref_R;
	while (1){
		vc >> imgr; if (imgr.empty())break;  // vc�� �о���� ������ img�� ���� //	flip(img, img, 1);// ���� ���ϰ� flip

		// for camera pose
		std::vector<cv::Point2f>  ref_points;


		bool found = findChessboardCorners(imgr, boardSize, ref_points);
		if (found){
			cv::solvePnP(obj_points, ref_points, K, dist_coeff, ref_rvec, ref_tvec);
			drawChessboardCorners(imgr, boardSize, cv::Mat(ref_points), found);
			Rodrigues(ref_rvec, ref_R); // reference point�� ���� ȸ����ȯ ��� �������
		} // camera pose ����(ref_rvec, ref_tvec)

		imshow("reference image", imgr);
		//������ ���Ͽ� ���������� ������.
		outputVideo << imgr;
		if (waitKey(10) == 32){
			ref_img = imgr;

			printf("ref_img complete;\n");
			break;
		}
	} // reference ���� ����
	destroyAllWindows();


	cv::Mat img_last;
	printf("new mode \n");
	cv::Mat sec_rvec, sec_tvec, sec_R;
	while (1){
		vc >> img; if (img.empty())break;  // vc�� �о���� ������ img�� ���� //	flip(img, img, 1);// ���� ���ϰ� flip
		// for camera pose
		std::vector<cv::Point2f>  sec_points;

		bool found = findChessboardCorners(img, boardSize, sec_points);
		if (found){
			cv::solvePnP(obj_points, sec_points, K, dist_coeff, sec_rvec, sec_tvec);
			drawChessboardCorners(img, boardSize, cv::Mat(sec_points), found);
			Rodrigues(sec_rvec, sec_R); // reference point�� ���� ȸ����ȯ ��� �������
		} // camera pose ����(ref_rvec, ref_tvec)

		// matching ���� ����(points1, points2��)
		cv::Mat img_m = img; // image for matching -> ��� ����ó���� �̹���
		std::vector<cv::Point2d> points1, points2;
		// feature ��Ī �κ� �ʿ�
		// Retrieve matching points
		cv::Ptr<cv::FeatureDetector> fdetector = cv::BRISK::create();	// Ư¡�����
		std::vector<cv::KeyPoint> keypoint1, keypoint2;
		cv::Mat descriptor1, descriptor2;
		fdetector->detectAndCompute(ref_img, cv::Mat(), keypoint1, descriptor1);	// keypoint, descriptor �������
		fdetector->detectAndCompute(img, cv::Mat(), keypoint2, descriptor2);
		cv::Ptr<cv::DescriptorMatcher> fmatcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
		std::vector<cv::DMatch> match;
		fmatcher->match(descriptor1, descriptor2, match);
		for (size_t i = 0; i < match.size(); i++)
		{
			points1.push_back(keypoint1.at(match.at(i).queryIdx).pt);
			points2.push_back(keypoint2.at(match.at(i).trainIdx).pt);
		}
		cv::Mat inlier_mask;
		cv::Mat H = cv::findHomography(points1, points2, inlier_mask, cv::RANSAC);

		cv::Mat merged;
		cv::warpPerspective(img, merged, H, cv::Size(img.cols * 2, img.rows));
		merged.colRange(0, img.cols) = img * 1; // Copy 




	// triangulation �غ���
		cv::Mat ref_Rt, sec_Rt;
		cv::hconcat(ref_R, ref_tvec, ref_Rt);
		cv::Mat P0 = K *ref_Rt;
		cv::hconcat(sec_R, sec_tvec, sec_Rt);
		//printf("ref_R size : %d, %d\n", ref_R.size().height, ref_R.size().width);
		//printf("ref_tvec size : %d, %d\n", ref_tvec.size().height, ref_tvec.size().width);
		//printf("sec_R size : %d, %d\n", sec_R.size().height, sec_R.size().width);
		//printf("sec_tvec size : %d, %d\n", sec_tvec.size().height, sec_tvec.size().width);
		cv::Mat P1 = K * sec_Rt;
		cv::triangulatePoints(P0, P1, points1, points2, X);
		X.row(0) = X.row(0) / X.row(3);
		X.row(1) = X.row(1) / X.row(3);
		X.row(2) = X.row(2) / X.row(3);
		X.row(3) = 1;

		/*
		cv::Mat F = cv::findFundamentalMat(points1, points2, cv::FM_8POINT);
		//cv::Mat K = (cv::Mat_<double>(3, 3) << camera_focal, 0, camera_center.x, 0, camera_focal, camera_center.y, 0, 0, 1);
		cv::Mat E = K.t() * F * K;
		cv::Mat R, t;
		cv::recoverPose(F, points1, points2, K, R, t);
		// Reconstruct 3D points of 'box.xyz' (triangulation)
		cv::Mat P0 = K * cv::Mat::eye(3, 4, CV_64F);
		cv::Mat Rt, X;
		cv::hconcat(R, t*board_cellsize, Rt);
		cv::Mat P1 = K * Rt;
		cv::triangulatePoints(P0, P1, points1, points2, X);
		X.row(0) = X.row(0) / X.row(3);
		X.row(1) = X.row(1) / X.row(3);
		X.row(2) = X.row(2) / X.row(3);
		X.row(3) = 1;
		*/
		

		// feature matching �ǽð� ȭ�� ���
		cv::Mat original, matched;
		cv::drawMatches(ref_img, keypoint1, img, keypoint2, match, matched, cv::Scalar::all(-1), cv::Scalar::all(-1), inlier_mask);
		cv::imshow("Feature matching and triangulation", matched);

		//������ ���Ͽ� ���������� ������.
		outputVideo << matched;
		// ���⼭ �����̽��� ������ triangulate ����� �����ش� �ϸ�Ƿ���?
		if (waitKey(10) == 32){
			printf("triangulation done\n");
			img_last = img;
			break;
		}
	} // feature matching ��
	destroyAllWindows();
	cout << X.size << endl;


	std::vector<cv::Point3d>  tri_points;		// triangulation ��� 3d ��ǥ
	printf("X Height : %d\nX Width : %d\n", X.size().height, X.size().width);
	for (int i = 0; i < X.size().width; i++){
		//tri_points.push_back(cv::Point3d(X.at<double>(0, i)* board_cellsize, X.at<double>(1, i)* board_cellsize, X.at<double>(2, i)* (board_cellsize)));
		tri_points.push_back(cv::Point3d(X.at<double>(0, i), X.at<double>(1, i), X.at<double>(2, i)));
	}
	//cout << typeid(X).name() << endl;
	printf("tri_points : %d\n", tri_points.size());

	// 3D reconstruction
	while (1){
		vc >> imgt; if (imgt.empty())break;  // vc�� �о���� ������ img�� ����
		std::vector<cv::Point2f>  w_points;
		std::vector<cv::Point2d>  w_points_;
		Mat w_R;
		cv::Mat w_rvec, w_tvec;
		bool found = findChessboardCorners(imgt, boardSize, w_points);
		if (found){
			cv::solvePnP(obj_points, w_points, K, dist_coeff, w_rvec, w_tvec);
			drawChessboardCorners(imgt, boardSize, cv::Mat(w_points), found);
			//	cout << norm(w_tvec) << endl;
			Rodrigues(w_rvec, w_R); // reference point�� ���� ȸ����ȯ ��� �������
			// 3D reconstruction projection
			cout << norm(w_tvec) << endl;
			projectPoints(tri_points, w_R, w_tvec, K, dist_coeff, w_points_);//* board_cellsize
			for (int n = 0; n < w_points_.size(); n++){
				circle(imgt, w_points_[n], 2, Scalar(255, 255, 255), 5);
			}

		} // camera pose ����

		/*
		// box projection - w_rvec, w_tvec Ȯ��
		Mat line_lower, line_upper;
		projectPoints(box_lower, w_rvec, w_tvec, K, dist_coeff, line_lower);
		projectPoints(box_upper, w_rvec, w_tvec, K, dist_coeff, line_upper);
		line_lower.reshape(1).convertTo(line_lower, CV_32S);
		line_upper.reshape(1).convertTo(line_upper, CV_32S);
		cv::polylines(imgt, line_lower, true, cv::Scalar(255, 0, 0), 2);
		for (int i = 0; i < line_lower.rows; i++){
		// cv::line(��ȭ��, ������,����,Scalar(Į��),(����׸�..? :�β�,����Ÿ��),shift)
		cv::line(imgt, cv::Point(line_lower.row(i)), cv::Point(line_upper.row(i)), cv::Scalar(0, 255, 255), 2); // �ذ� �� ����
		//	cout << "�̰� ����lower" << line_lower.row(i) << "�̰� ���� upper" <<line_upper.row(i)<< endl;
		}
		cv::polylines(imgt, line_upper, true, cv::Scalar(0, 0, 255), 2);
		*/


		imshow("3D reconstruction", imgt);
		outputVideo << imgt;
		if (waitKey(10) == 27)break;
	}
	destroyAllWindows();



	return 0;
}
