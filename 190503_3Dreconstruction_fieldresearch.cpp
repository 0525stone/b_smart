#include "opencv_all.hpp"

#include <iostream>  
#include "opencv2/opencv.hpp"  
#include <typeinfo>

using namespace cv;
using namespace std;

int main(void)
{
	// VideoCapture로 Video영상을 불러옴
	VideoCapture vc(0); if (!vc.isOpened())return 0; // connection failed

	// 필요변수 선언 캘리브레이션 결과
	cv::Mat K = (cv::Mat_<double>(3, 3) << 640.3783, 0, 316.6393, 0, 679.3930, 227.904493, 0, 0, 1);
	//	cv::Mat dist_coeff = (cv::Mat_<double>(4, 1) << 0.004628, -0.044073, 0.008047, 0.001567);
	cv::Mat dist_coeff = (cv::Mat_<double>(4, 1) << 0.056178, -0.213159, -0.029619, 0.014168);
	Mat img, imgr, imgt;  // 영상 받아줄 Matrix 선언
	Size boardSize(4, 7); // chessboard 사이즈 (체크사이 격자 기준)
	float board_cellsize = 26.0;

	// 공간상의 큐브 점들 정의 (projection 동작 확인용)
	std::vector<cv::Point3d> box_lower, box_upper;
	box_lower.push_back(cv::Point3d(3 * board_cellsize, 2 * board_cellsize, 0));
	box_lower.push_back(cv::Point3d(5 * board_cellsize, 2 * board_cellsize, 0));
	box_lower.push_back(cv::Point3d(5 * board_cellsize, 4 * board_cellsize, 0));
	box_lower.push_back(cv::Point3d(3 * board_cellsize, 4 * board_cellsize, 0));
	box_upper.push_back(cv::Point3d(3 * board_cellsize, 2 * board_cellsize, -2 * board_cellsize));
	box_upper.push_back(cv::Point3d(5 * board_cellsize, 2 * board_cellsize, -2 * board_cellsize));
	box_upper.push_back(cv::Point3d(5 * board_cellsize, 4 * board_cellsize, -2 * board_cellsize));
	box_upper.push_back(cv::Point3d(3 * board_cellsize, 4 * board_cellsize, -2 * board_cellsize));

	//웹캠에서 캡쳐되는 이미지 크기를 가져옴
	Size size = Size((int)vc.get(CAP_PROP_FRAME_WIDTH),
		(int)vc.get(CAP_PROP_FRAME_HEIGHT));

	//파일로 동영상을 저장하기 위한 준비  
	VideoWriter outputVideo;
	outputVideo.open("ouput.avi", VideoWriter::fourcc('X', 'V', 'I', 'D'),
		15, size, true);
	if (!outputVideo.isOpened())
	{
		cout << "동영상을 저장하기 위한 초기화 작업 중 에러 발생" << endl;
		return 1;
	}

	cv::Mat ref_img, X;
	// 이미지 처리 / 영상 시작

	// 변수 for camera pose
	std::vector<cv::Point3f>  obj_points;
	for (int r = 0; r < boardSize.height; r++)
	for (int c = 0; c < boardSize.width; c++)
		obj_points.push_back(cv::Point3d(board_cellsize * c, board_cellsize * r, 0));

	cv::Mat ref_rvec, ref_tvec, ref_R;
	while (1){
		vc >> imgr; if (imgr.empty())break;  // vc로 읽어들인 영상을 img로 해줌 //	flip(img, img, 1);// 보기 편하게 flip

		// for camera pose
		std::vector<cv::Point2f>  ref_points;


		bool found = findChessboardCorners(imgr, boardSize, ref_points);
		if (found){
			cv::solvePnP(obj_points, ref_points, K, dist_coeff, ref_rvec, ref_tvec);
			drawChessboardCorners(imgr, boardSize, cv::Mat(ref_points), found);
			Rodrigues(ref_rvec, ref_R); // reference point에 대한 회전변환 행렬 만들어줌
		} // camera pose 구함(ref_rvec, ref_tvec)

		imshow("reference image", imgr);
		//동영상 파일에 한프레임을 저장함.
		outputVideo << imgr;
		if (waitKey(10) == 32){
			ref_img = imgr;

			printf("ref_img complete;\n");
			break;
		}
	} // reference 영상 저장
	destroyAllWindows();


	cv::Mat img_last;
	printf("new mode \n");
	cv::Mat sec_rvec, sec_tvec, sec_R;
	while (1){
		vc >> img; if (img.empty())break;  // vc로 읽어들인 영상을 img로 해줌 //	flip(img, img, 1);// 보기 편하게 flip
		// for camera pose
		std::vector<cv::Point2f>  sec_points;

		bool found = findChessboardCorners(img, boardSize, sec_points);
		if (found){
			cv::solvePnP(obj_points, sec_points, K, dist_coeff, sec_rvec, sec_tvec);
			drawChessboardCorners(img, boardSize, cv::Mat(sec_points), found);
			Rodrigues(sec_rvec, sec_R); // reference point에 대한 회전변환 행렬 만들어줌
		} // camera pose 구함(ref_rvec, ref_tvec)

		// matching 점들 추출(points1, points2로)
		cv::Mat img_m = img; // image for matching -> 얘는 영상처리용 이미지
		std::vector<cv::Point2d> points1, points2;
		// feature 매칭 부분 필요
		// Retrieve matching points
		cv::Ptr<cv::FeatureDetector> fdetector = cv::BRISK::create();	// 특징추출기
		std::vector<cv::KeyPoint> keypoint1, keypoint2;
		cv::Mat descriptor1, descriptor2;
		fdetector->detectAndCompute(ref_img, cv::Mat(), keypoint1, descriptor1);	// keypoint, descriptor 만들어짐
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




	// triangulation 해보자
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
		

		// feature matching 실시간 화면 출력
		cv::Mat original, matched;
		cv::drawMatches(ref_img, keypoint1, img, keypoint2, match, matched, cv::Scalar::all(-1), cv::Scalar::all(-1), inlier_mask);
		cv::imshow("Feature matching and triangulation", matched);

		//동영상 파일에 한프레임을 저장함.
		outputVideo << matched;
		// 여기서 스페이스바 누르면 triangulate 결과를 보여준다 하면되려나?
		if (waitKey(10) == 32){
			printf("triangulation done\n");
			img_last = img;
			break;
		}
	} // feature matching 끝
	destroyAllWindows();
	cout << X.size << endl;


	std::vector<cv::Point3d>  tri_points;		// triangulation 결과 3d 좌표
	printf("X Height : %d\nX Width : %d\n", X.size().height, X.size().width);
	for (int i = 0; i < X.size().width; i++){
		//tri_points.push_back(cv::Point3d(X.at<double>(0, i)* board_cellsize, X.at<double>(1, i)* board_cellsize, X.at<double>(2, i)* (board_cellsize)));
		tri_points.push_back(cv::Point3d(X.at<double>(0, i), X.at<double>(1, i), X.at<double>(2, i)));
	}
	//cout << typeid(X).name() << endl;
	printf("tri_points : %d\n", tri_points.size());

	// 3D reconstruction
	while (1){
		vc >> imgt; if (imgt.empty())break;  // vc로 읽어들인 영상을 img로 해줌
		std::vector<cv::Point2f>  w_points;
		std::vector<cv::Point2d>  w_points_;
		Mat w_R;
		cv::Mat w_rvec, w_tvec;
		bool found = findChessboardCorners(imgt, boardSize, w_points);
		if (found){
			cv::solvePnP(obj_points, w_points, K, dist_coeff, w_rvec, w_tvec);
			drawChessboardCorners(imgt, boardSize, cv::Mat(w_points), found);
			//	cout << norm(w_tvec) << endl;
			Rodrigues(w_rvec, w_R); // reference point에 대한 회전변환 행렬 만들어줌
			// 3D reconstruction projection
			cout << norm(w_tvec) << endl;
			projectPoints(tri_points, w_R, w_tvec, K, dist_coeff, w_points_);//* board_cellsize
			for (int n = 0; n < w_points_.size(); n++){
				circle(imgt, w_points_[n], 2, Scalar(255, 255, 255), 5);
			}

		} // camera pose 구함

		/*
		// box projection - w_rvec, w_tvec 확인
		Mat line_lower, line_upper;
		projectPoints(box_lower, w_rvec, w_tvec, K, dist_coeff, line_lower);
		projectPoints(box_upper, w_rvec, w_tvec, K, dist_coeff, line_upper);
		line_lower.reshape(1).convertTo(line_lower, CV_32S);
		line_upper.reshape(1).convertTo(line_upper, CV_32S);
		cv::polylines(imgt, line_lower, true, cv::Scalar(255, 0, 0), 2);
		for (int i = 0; i < line_lower.rows; i++){
		// cv::line(도화지, 시작점,끝점,Scalar(칼라),(없어도그만..? :두께,라인타입),shift)
		cv::line(imgt, cv::Point(line_lower.row(i)), cv::Point(line_upper.row(i)), cv::Scalar(0, 255, 255), 2); // 밑과 위 연결
		//	cout << "이게 라인lower" << line_lower.row(i) << "이건 라인 upper" <<line_upper.row(i)<< endl;
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
