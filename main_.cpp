#include <iostream>  
#include "opencv2/opencv.hpp"  
#include <typeinfo>

using namespace cv;
using namespace std;

int main()
{
	// VideoCapture�� Video������ �ҷ���
	VideoCapture vc(0); if (!vc.isOpened())return 0; // connection failed
	vc.set(CV_CAP_PROP_FRAME_WIDTH, 640);  // CAP_PROP�� ����...
	vc.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	// �ʿ亯�� ����
	// Ķ���극�̼� ���
	cv::Mat K = (cv::Mat_<double>(3, 3) << 634.562348, 0, 305.212497, 0, 634.205282, 256.234517, 0, 0, 1);
	cv::Mat dist_coeff = (cv::Mat_<double>(4, 1) << 0.004628, -0.044073, 0.008047, 0.001567);
	Mat img;  // ���� �޾��� Matrix ����
	Size boardSize(4, 7); // chessboard ������ (üũ���� ���� ����)
	float board_cellsize = 26.0;
	// �������� ť�� ���� ����
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


	//��ķ���� ĸ�ĵǴ� �̹��� ũ�⸦ ������
	Size size = Size((int)vc.get(CAP_PROP_FRAME_WIDTH),
		(int)vc.get(CAP_PROP_FRAME_HEIGHT));

	//���Ϸ� �������� �����ϱ� ���� �غ�  
	VideoWriter outputVideo;
	outputVideo.open("ouput.avi", VideoWriter::fourcc('X', 'V', 'I', 'D'),
		30, size, true);
	if (!outputVideo.isOpened())
	{
		cout << "�������� �����ϱ� ���� �ʱ�ȭ �۾� �� ���� �߻�" << endl;
		return 1;
	}


	// �̹��� ó�� / ���� ����
	while (1){
		vc >> img; if (img.empty())break;  // vc�� �о���� ������ img�� ����
		flip(img, img, 1);				   // ���� ���ϰ� flip

		// vector�� �̹��� ��ǥ �޾��� ���� ����
		vector<std::vector<cv::Point2f> > imagePoints(1);
		vector<std::vector<cv::Point3f> > objectPoints(1);
		Mat rvec, tvec;
		bool found = findChessboardCorners(img, boardSize, imagePoints[0]);
		//printf("Hello?");
		for (int r = 0; r < boardSize.height; r++)
			for (int c = 0; c < boardSize.width; c++)
				// push_back (vector ���� �ڷ� �߰��ϴ� ��ɾ�)
				objectPoints[0].push_back(cv::Point3d(board_cellsize * c, board_cellsize * r, 0));
		

		if (found){ // ü�����带 ã���� �ϴ� �͵�
			// solvePnP�� rvec, tvec ã��!
			cv::solvePnP(objectPoints[0], imagePoints[0], K, dist_coeff, rvec, tvec); // objPoints�� 3��������
			drawChessboardCorners(img, boardSize, cv::Mat(imagePoints[0]), found);
			
			std::vector<cv::Point2f>line_lower, line_upper;
		//	Mat line_lower, line_upper ;   // �̰� ���߿� double(3,4)�÷� ���� �ʾƵ� �Ǵ��� Ȯ���Ҷ� ���°�
			Mat R;
			Rodrigues(rvec, R);	//Rodigues() : rvec���� ȸ����ȯ ��Ʈ���� �������
		//	Mat R_inv = R.inv();
	//		Mat R = R.inv();
			for (int r = 0; r < box_lower.size(); r++) {
				cv::Mat box_row = (cv::Mat_<double>(3, 1) << box_lower[r].x, box_lower[r].y, box_lower[r].z);
				cv::Mat box_row_up = (cv::Mat_<double>(3, 1) << box_upper[r].x, box_upper[r].y, box_upper[r].z);
				// �ε帮�Խ� ����̶� tvec�� ���� 3x1 matrix����......
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
				//cv::line(��ȭ��, ������,����,scalar(Į��),(����׸�..? :�β�,����Ÿ��),shift)
				cv::line(img, point1, point2, cv::Scalar(0, 255, 255), 2); // �ذ� �� ����
			}
			for (int i = 0; i < line_upper.size(); i++){
				Point point1 = (line_upper[i % 4]);
				Point point2 = (line_upper[(i + 1) % 4]);
				cv::line(img, point1, point2, cv::Scalar(255, 255, 0), 2);
			}

		}
		//drawChessboardCorners(img, boardSize, cv::Mat(imagePoints[0]), found);
		imshow("cam", img);
		//������ ���Ͽ� ���������� ������.  
		outputVideo << img;
		if (waitKey(10) == 27)break;
	}
	destroyAllWindows();
	return 0;
}

