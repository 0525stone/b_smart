#include <iostream>  
#include <typeinfo>
#include "opencv2/opencv.hpp"  

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
		//	cout << typeid(tvec).name() << endl;
			Mat line_lower, line_upper;
			//  cv::projectPoints(투영할 공간좌표, rVec, tVec, K(내부행렬), distCoeffs, 결과값.투영된좌표)
			projectPoints(box_lower, rvec, tvec, K, dist_coeff, line_lower);
			projectPoints(box_upper, rvec, tvec, K, dist_coeff, line_upper);
			// reshape(숫자) : 채널을 숫자만큼 똑같이 복사, converTo(결과변수,자료형):polyline을 위해 CV_32S로 바꿔줌. 
			line_lower.reshape(1).convertTo(line_lower, CV_32S); // Change 4 x 1 matrix (CV_64FC2) to 4 x 2 matrix (CV_32SC1)
			line_upper.reshape(1).convertTo(line_upper, CV_32S); // Because 'cv::polylines()' only accepts 'CV_32S' depth.
			//cout << line_lower << endl;
			//break;
			// polylines(도화지,좌표들,t/f 닫히냐마냐, 칼라(스칼라로), shift)
			cv::polylines(img, line_lower, true, cv::Scalar(255, 0, 0), 2);
			for (int i = 0; i < line_lower.rows; i++){
				// cv::line(도화지, 시작점,끝점,Scalar(칼라),(없어도그만..? :두께,라인타입),shift)
				cv::line(img, cv::Point(line_lower.row(i)), cv::Point(line_upper.row(i)), cv::Scalar(0, 255, 255), 2); // 밑과 위 연결
				cout << "이게 라인lower" << line_lower.row(i) << "이건 라인 upper" <<line_upper.row(i)<< endl;
			}
			cv::polylines(img, line_upper, true, cv::Scalar(0, 0, 255), 2);

		}
		
		imshow("cam", img);
		//동영상 파일에 한프레임을 저장함.  
		outputVideo << img;
		if (waitKey(10) == 27)break;
	}
	destroyAllWindows();
	return 0;
}
