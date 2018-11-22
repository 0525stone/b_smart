#include <iostream>  
#include "opencv2/opencv.hpp"  
#include <typeinfo>

using namespace cv;
using namespace std;

int main()
{
	// VideoCapture로 Video영상을 불러옴
	VideoCapture vc(0); if (!vc.isOpened())return 0; // connection failed
	vc.set(CV_CAP_PROP_FRAME_WIDTH, 640);  // CAP_PROP는 뭔뜻...
	vc.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	// 필요변수 선언
	// 캘리브레이션 결과
	cv::Mat K = (cv::Mat_<double>(3, 3) << 634.562348, 0, 305.212497, 0, 634.205282, 256.234517, 0, 0, 1);
	cv::Mat dist_coeff = (cv::Mat_<double>(4, 1) << 0.004628, -0.044073, 0.008047, 0.001567);
	Mat img;  // 영상 받아줄 Matrix 선언
	Size boardSize(4, 7); // chessboard 사이즈 (체크사이 격자 기준)
	float board_cellsize = 26.0;
	// 공간상의 큐브 점들 정의
	std::vector<cv::Point3d> box_lower, box_upper;
	box_lower.push_back(cv::Point3d(3 * board_cellsize, 2 * board_cellsize, 0));
	box_lower.push_back(cv::Point3d(5 * board_cellsize, 2 * board_cellsize, 0));
	box_lower.push_back(cv::Point3d(5 * board_cellsize, 4 * board_cellsize, 0));
	box_lower.push_back(cv::Point3d(3 * board_cellsize, 4 * board_cellsize, 0));
	box_upper.push_back(cv::Point3d(3 * board_cellsize, 2 * board_cellsize, -2 * board_cellsize));
	box_upper.push_back(cv::Point3d(5 * board_cellsize, 2 * board_cellsize, -2 * board_cellsize));
	box_upper.push_back(cv::Point3d(5 * board_cellsize, 4 * board_cellsize, -2 * board_cellsize));
	box_upper.push_back(cv::Point3d(3 * board_cellsize, 4 * board_cellsize, -2 * board_cellsize));
	//box_lower.push_back(cv::Point3d(0 * board_cellsize, 2 * board_cellsize, 0));
	//box_lower.push_back(cv::Point3d(2 * board_cellsize, 2 * board_cellsize, 0));
	//box_lower.push_back(cv::Point3d(2 * board_cellsize, 4 * board_cellsize, 0));
	//box_lower.push_back(cv::Point3d(0 * board_cellsize, 4 * board_cellsize, 0));
	//box_upper.push_back(cv::Point3d(0 * board_cellsize, 2 * board_cellsize, -2 * board_cellsize));
	//box_upper.push_back(cv::Point3d(2 * board_cellsize, 2 * board_cellsize, -2 * board_cellsize));
	//box_upper.push_back(cv::Point3d(2 * board_cellsize, 4 * board_cellsize, -2 * board_cellsize));
	//box_upper.push_back(cv::Point3d(0 * board_cellsize, 4 * board_cellsize, -2 * board_cellsize));


	//웹캠에서 캡쳐되는 이미지 크기를 가져옴
	Size size = Size((int)vc.get(CAP_PROP_FRAME_WIDTH),
		(int)vc.get(CAP_PROP_FRAME_HEIGHT));

	//파일로 동영상을 저장하기 위한 준비  
	VideoWriter outputVideo;
	outputVideo.open("ouput.avi", VideoWriter::fourcc('X', 'V', 'I', 'D'),
		30, size, true);
	if (!outputVideo.isOpened())
	{
		cout << "동영상을 저장하기 위한 초기화 작업 중 에러 발생" << endl;
		return 1;
	}


	// 이미지 처리 / 영상 시작
	while (1){
		vc >> img; if (img.empty())break;  // vc로 읽어들인 영상을 img로 해줌
		flip(img, img, 1);				   // 보기 편하게 flip

		// vector로 이미지 좌표 받아줄 변수 선언
		vector<std::vector<cv::Point2f> > imagePoints(1);
		vector<std::vector<cv::Point3f> > objectPoints(1);
		Mat rvec, tvec;
		bool found = findChessboardCorners(img, boardSize, imagePoints[0]);
		//printf("Hello?");
		for (int r = 0; r < boardSize.height; r++)
			for (int c = 0; c < boardSize.width; c++)
				// push_back (vector 형에 자료 추가하는 명령어)
				objectPoints[0].push_back(cv::Point3d(board_cellsize * c, board_cellsize * r, 0));
		

		if (found){ // 체스보드를 찾으면 하는 것들
			// solvePnP로 rvec, tvec 찾기!
			cv::solvePnP(objectPoints[0], imagePoints[0], K, dist_coeff, rvec, tvec); // objPoints는 3차원기준
			drawChessboardCorners(img, boardSize, cv::Mat(imagePoints[0]), found);
			
			std::vector<cv::Point2f>line_lower, line_upper;
		//	Mat line_lower, line_upper ;   // 이건 나중에 double(3,4)꼴로 하지 않아도 되는지 확인할때 쓰는거
			Mat R;
			Rodrigues(rvec, R);	//Rodigues() : rvec으로 회전변환 매트릭스 만들어줌
		//	Mat R_inv = R.inv();
	//		Mat R = R.inv();
			for (int r = 0; r < box_lower.size(); r++) {
				cv::Mat box_row = (cv::Mat_<double>(3, 1) << box_lower[r].x, box_lower[r].y, box_lower[r].z);
				cv::Mat box_row_up = (cv::Mat_<double>(3, 1) << box_upper[r].x, box_upper[r].y, box_upper[r].z);
				// 로드리게스 행렬이랑 tvec의 곱은 3x1 matrix였다......
				box_row = K*(R * box_row + tvec);
				box_row_up = K*(R * box_row_up + tvec);
				box_row = box_row / box_row.at<double>(2,0);
				box_row_up = box_row_up / box_row_up.at<double>(2, 0);
				line_lower.push_back(cv::Point2f((int)box_row.at<double>(0, 0), (int)box_row.at<double>(1, 0)));
				line_upper.push_back(cv::Point2f((int)box_row_up.at<double>(0, 0), (int)box_row_up.at<double>(1, 0)));				
			}

		//	polylines(img, line_lower, true, cv::Scalar(255, 0, 0), 2);
			for (int i = 0; i < line_lower.size(); i++){
				Point point1 = (line_lower[i%4]);
				Point point2 = (line_lower[(i+1)%4]);
				cv::line(img, point1, point2, cv::Scalar(0, 255, 0), 2); 
			}
			for (int i = 0; i < line_lower.size(); i++){
				Point point1 = Point(line_lower[i].x, line_lower[i].y);
				Point point2 = (line_upper[i]);
				//cv::line(도화지, 시작점,끝점,scalar(칼라),(없어도그만..? :두께,라인타입),shift)
				cv::line(img, point1, point2, cv::Scalar(0, 255, 255), 2); // 밑과 위 연결
			}
			for (int i = 0; i < line_upper.size(); i++){
				Point point1 = (line_upper[i % 4]);
				Point point2 = (line_upper[(i + 1) % 4]);
				cv::line(img, point1, point2, cv::Scalar(255, 255, 0), 2);
			}

		}
		//drawChessboardCorners(img, boardSize, cv::Mat(imagePoints[0]), found);
		imshow("cam", img);
		//동영상 파일에 한프레임을 저장함.  
		outputVideo << img;
		if (waitKey(10) == 27)break;
	}
	destroyAllWindows();
	return 0;
}

